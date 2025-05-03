import os
import logging
import math
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from accelerate import Accelerator

import torchvision.transforms as transforms
from tqdm import tqdm

from diffusers import DDPMScheduler
from src.build import (
    build_unet,
    build_style_encoder,
    build_content_encoder,
    build_shading_encoder,
    build_background_encoder,
    build_scr,
    build_ddpm_scheduler,
)
from src.model import FontDiffuserModel
from src.criterion import (
    ContentPerceptualLoss,
    ShadingTextureLoss,
    BackgroundPerceptualLoss,
)
from dataset.font_dataset import FontDataset
from configs.fontdiffuser import get_parser

try:
    import lpips
except ImportError:
    lpips = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed(args):
    """Initialize distributed training if applicable"""
    if args.local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    return args.local_rank != -1

def get_transforms(args):
    """Define transforms for dataset"""
    content_transform = transforms.Compose([
        transforms.Resize((args.content_image_size, args.content_image_size),
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    style_transform = transforms.Compose([
        transforms.Resize((args.style_image_size, args.style_image_size),
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution),
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    scr_transform = transforms.Compose([
        transforms.Resize((args.scr_image_size, args.scr_image_size),
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return [content_transform, style_transform, target_transform, scr_transform]

def get_dataloaders(args, transforms):
    """Create train and validation dataloaders"""
    train_dataset = FontDataset(
        args=args,
        phase="train",
        transforms=transforms,
        scr=args.phase_2,
    )
    val_dataset = FontDataset(
        args=args,
        phase="val",
        transforms=transforms,
        scr=args.phase_2,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader

def initialize_model_and_optimizer(args, accelerator):
    """Initialize model, scheduler, and optimizer"""
    unet = build_unet(args)
    style_encoder = build_style_encoder(args)
    content_encoder = build_content_encoder(args)
    shading_encoder = build_shading_encoder(args)
    background_encoder = build_background_encoder(args)
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        shading_encoder=shading_encoder,
        background_encoder=background_encoder,
    )
    scheduler = build_ddpm_scheduler(args)

    # Load phase 1 checkpoint if in phase 2
    if args.phase_2 and args.phase_1_ckpt_dir:
        checkpoint = torch.load(os.path.join(args.phase_1_ckpt_dir, "model.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded phase 1 checkpoint")

    # Initialize SCR module for phase 2
    scr = None
    if args.phase_2:
        scr = build_scr(args)
        if args.scr_ckpt_path and os.path.exists(args.scr_ckpt_path):
            scr.load_state_dict(torch.load(args.scr_ckpt_path, map_location="cpu"))
            logger.info("Loaded SCR checkpoint")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_train_steps,
        eta_min=args.learning_rate * 0.1,
    )
    if args.lr_warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.lr_warmup_steps,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[args.lr_warmup_steps],
        )

    # Prepare with Accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    if scr is not None:
        scr = accelerator.prepare(scr)

    return model, scheduler, optimizer, lr_scheduler, scr

def train_step(args, model, scheduler, data, accelerator, scaler, scr=None):
    """Perform one training step"""
    content_images = data["content_image"].to(accelerator.device)
    style_images = data["style_image"].to(accelerator.device)
    target_images = data["target_image"].to(accelerator.device)
    shading_images = data["shading_image"].to(accelerator.device)
    background_images = data["background_image"].to(accelerator.device)
    neg_images = data.get("neg_images", None)
    batch_size = content_images.shape[0]

    # Sample timesteps
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device)
    noise = torch.randn_like(target_images)
    x_t = scheduler.add_noise(target_images, noise, timesteps)

    # Classifier-free guidance: randomly drop inputs
    if args.drop_prob > 0:
        drop_mask = torch.rand(batch_size, device=accelerator.device) < args.drop_prob
        content_images[drop_mask] = torch.zeros_like(content_images[drop_mask])
        style_images[drop_mask] = torch.zeros_like(style_images[drop_mask])
        shading_images[drop_mask] = torch.zeros_like(shading_images[drop_mask])
        background_images[drop_mask] = torch.zeros_like(background_images[drop_mask])

    # Forward pass
    with autocast():
        noise_pred, offset_out_sum = model(
            x_t,
            timesteps,
            style_images,
            content_images,
            shading_images,
            background_images,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
        )
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # Predict x_0 and compute perceptual losses
        pred_x0 = scheduler.step(noise_pred, timesteps, x_t).pred_original_sample
        pred_x0 = (pred_x0 + 1) / 2  # [-1, 1] to [0, 1] for VGG/LPIPS
        target_images_norm = (target_images + 1) / 2  # [-1, 1] to [0, 1]

        # Content perceptual loss
        content_loss_fn = ContentPerceptualLoss()
        content_loss = content_loss_fn.calculate_loss(pred_x0, target_images_norm, accelerator.device)

        # Shading texture loss
        shading_loss_fn = ShadingTextureLoss()
        shading_loss = shading_loss_fn.calculate_loss(pred_x0, target_images_norm, accelerator.device)

        # Background perceptual loss
        if args.use_lpips and lpips is not None:
            lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)
            pred_x0_lpips = pred_x0 * 2 - 1  # [0, 1] to [-1, 1] for LPIPS
            target_images_lpips = target_images_norm * 2 - 1
            background_loss = lpips_loss_fn(pred_x0_lpips, target_images_lpips).mean()
        else:
            background_loss_fn = BackgroundPerceptualLoss()
            background_loss = background_loss_fn.calculate_loss(pred_x0, target_images_norm, accelerator.device)

        # Offset loss
        offset_loss = offset_out_sum.abs().mean()

        # SCR loss (phase 2)
        scr_loss = torch.tensor(0.0, device=accelerator.device)
        if args.phase_2 and scr is not None and neg_images is not None:
            neg_images = neg_images.to(accelerator.device)
            # Resize images for SCR (96x96)
            scr_transform = transforms.Compose([
                transforms.Resize((args.scr_image_size, args.scr_image_size)),
                transforms.Normalize([0.5], [0.5]),
            ])
            content_images_scr = torch.stack([scr_transform(img) for img in content_images])
            style_images_scr = torch.stack([scr_transform(img) for img in style_images])
            neg_images_scr = torch.stack([scr_transform(img) for img in neg_images.view(-1, 3, args.content_image_size, args.content_image_size)])
            neg_images_scr = neg_images_scr.view(batch_size, args.num_neg, 3, args.scr_image_size, args.scr_image_size)
            
            sample_s, pos_s, neg_s = scr(content_images_scr, style_images_scr, neg_images_scr)
            scr_loss = scr.calculate_nce_loss(sample_s, pos_s, neg_s)

        # Total loss
        total_loss = (
            diffusion_loss +
            args.perceptual_coefficient * content_loss +
            args.shading_coefficient * shading_loss +
            args.background_coefficient * background_loss +
            args.offset_coefficient * offset_loss +
            args.sc_coefficient * scr_loss
        )

    # Backward pass
    accelerator.backward(total_loss, scaler=scaler)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    return {
        "total_loss": total_loss.item(),
        "diffusion_loss": diffusion_loss.item(),
        "content_loss": content_loss.item(),
        "shading_loss": shading_loss.item(),
        "background_loss": background_loss.item(),
        "offset_loss": offset_loss.item(),
        "scr_loss": scr_loss.item(),
    }

def validate(args, model, scheduler, dataloader, accelerator):
    """Perform validation and return average losses"""
    model.eval()
    total_losses = {
        "total_loss": 0.0,
        "diffusion_loss": 0.0,
        "content_loss": 0.0,
        "shading_loss": 0.0,
        "background_loss": 0.0,
        "offset_loss": 0.0,
        "scr_loss": 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for data in dataloader:
            content_images = data["content_image"].to(accelerator.device)
            style_images = data["style_image"].to(accelerator.device)
            target_images = data["target_image"].to(accelerator.device)
            shading_images = data["shading_image"].to(accelerator.device)
            background_images = data["background_image"].to(accelerator.device)
            batch_size = content_images.shape[0]

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device)
            noise = torch.randn_like(target_images)
            x_t = scheduler.add_noise(target_images, noise, timesteps)

            noise_pred, offset_out_sum = model(
                x_t,
                timesteps,
                style_images,
                content_images,
                shading_images,
                background_images,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
            )
            diffusion_loss = F.mse_loss(noise_pred, noise)

            pred_x0 = scheduler.step(noise_pred, timesteps, x_t).pred_original_sample
            pred_x0 = (pred_x0 + 1) / 2
            target_images_norm = (target_images + 1) / 2

            content_loss_fn = ContentPerceptualLoss()
            content_loss = content_loss_fn.calculate_loss(pred_x0, target_images_norm, accelerator.device)

            shading_loss_fn = ShadingTextureLoss()
            shading_loss = shading_loss_fn.calculate_loss(pred_x0, target_images_norm, accelerator.device)

            if args.use_lpips and lpips is not None:
                lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)
                pred_x0_lpips = pred_x0 * 2 - 1
                target_images_lpips = target_images_norm * 2 - 1
                background_loss = lpips_loss_fn(pred_x0_lpips, target_images_lpips).mean()
            else:
                background_loss_fn = BackgroundPerceptualLoss()
                background_loss = background_loss_fn.calculate_loss(pred_x0, target_images_norm, accelerator.device)

            offset_loss = offset_out_sum.abs().mean()

            total_loss = (
                diffusion_loss +
                args.perceptual_coefficient * content_loss +
                args.shading_coefficient * shading_loss +
                args.background_coefficient * background_loss +
                args.offset_coefficient * offset_loss
            )

            total_losses["total_loss"] += total_loss.item()
            total_losses["diffusion_loss"] += diffusion_loss.item()
            total_losses["content_loss"] += content_loss.item()
            total_losses["shading_loss"] += shading_loss.item()
            total_losses["background_loss"] += background_loss.item()
            total_losses["offset_loss"] += offset_loss.item()
            num_batches += 1

    model.train()
    return {k: v / num_batches for k, v in total_losses.items()}

def main():
    args = get_parser().parse_args()
    is_distributed = setup_distributed(args)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )

    # Initialize TensorBoard
    if accelerator.is_main_process:
        log_dir = os.path.join(args.logging_dir, args.experience_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs("ckpt", exist_ok=True)

    # Initialize dataloaders
    transforms = get_transforms(args)
    train_dataloader, val_dataloader = get_dataloaders(args, transforms)
    train_dataloader = accelerator.prepare(train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)

    # Initialize model and optimizer
    model, scheduler, optimizer, lr_scheduler, scr = initialize_model_and_optimizer(args, accelerator)
    scaler = GradScaler() if args.mixed_precision == "fp16" else None

    # Enable gradient checkpointing
    model.unet.gradient_checkpointing = True

    # Training loop
    global_step = 0
    for epoch in range(math.ceil(args.max_train_steps / len(train_dataloader))):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
        for data in progress_bar:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(model):
                losses = train_step(args, model, scheduler, data, accelerator, scaler, scr)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Log losses
            if global_step % args.log_interval == 0 and accelerator.is_main_process:
                for key, value in losses.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

            # Validation and checkpointing
            if global_step % args.ckpt_interval == 0 and global_step > 0:
                val_losses = validate(args, model, scheduler, val_dataloader, accelerator)
                if accelerator.is_main_process:
                    for key, value in val_losses.items():
                        writer.add_scalar(f"val/{key}", value, global_step)
                    logger.info(f"Step {global_step}: Val Losses - {', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()])}")

                    # Save checkpoint
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "global_step": global_step,
                    }
                    checkpoint_path = os.path.join("ckpt", f"checkpoint_{global_step}.pth")
                    accelerator.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint at {checkpoint_path}")

            global_step += 1

        if global_step >= args.max_train_steps:
            break

    # Final checkpoint
    if accelerator.is_main_process:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "global_step": global_step,
        }
        accelerator.save(checkpoint, os.path.join("ckpt", "model.pth"))
        logger.info("Saved final checkpoint")

    if writer is not None:
        writer.close()
    accelerator.end_training()

if __name__ == "__main__":
    main()