import os
import cv2
import time
import random
import logging
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
    build_shading_encoder,
    build_background_encoder,
)
from utils import (
    ttf2im,
    load_ttf,
    is_char_in_font,
    save_args_to_yaml,
    save_single_image,
    save_image_with_content_style,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False)
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--shading_image_path", type=str, default=None)
    parser.add_argument("--background_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    return args

def validate_paths(args):
    """Validate input paths against TE141K structure"""
    if not args.demo:
        if args.character_input:
            if not args.content_character:
                raise ValueError("content_character must be provided with --character_input")
        else:
            if not args.content_image_path or not os.path.exists(args.content_image_path):
                raise ValueError(f"Invalid or missing content_image_path: {args.content_image_path}")
        if not args.style_image_path or not os.path.exists(args.style_image_path):
            raise ValueError(f"Invalid or missing style_image_path: {args.style_image_path}")
        if not args.shading_image_path or not os.path.exists(args.shading_image_path):
            logger.warning(f"Shading image missing: {args.shading_image_path}, using blank image")
            args.shading_image_path = None
        if not args.background_image_path or not os.path.exists(args.background_image_path):
            logger.warning(f"Background image missing: {args.background_image_path}, using blank image")
            args.background_image_path = None

def char_to_index(char):
    """Map character to index (placeholder; replace with metadata)"""
    # Assume CJK Unicode range starting at U+4E00
    cjk_start = 0x4E00
    index = ord(char) - cjk_start
    if index < 0 or index > 10000:  # Arbitrary limit
        logger.warning(f"Character {char} outside expected range, using index 0")
        index = 0
    return str(index)

def image_process(args, content_image=None, style_image=None, shading_image=None, background_image=None):
    validate_paths(args)
    
    if not args.demo:
        if args.character_input:
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                logger.error(f"Character {args.content_character} not supported in {args.ttf_path}")
                return None, None, None, None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil = None
        style_image = Image.open(args.style_image_path).convert('RGB')
        shading_image = Image.open(args.shading_image_path).convert('RGB') if args.shading_image_path else \
                        Image.new('RGB', args.content_image_size)
        background_image = Image.open(args.background_image_path).convert('RGB') if args.background_image_path else \
                          Image.new('RGB', args.content_image_size)
    else:
        if not style_image:
            raise ValueError("Style image must be provided in demo mode")
        if args.character_input:
            if not args.content_character or not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                logger.error(f"Character {args.content_character} not supported in {args.ttf_path}")
                return None, None, None, None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            if not content_image:
                raise ValueError("Content image must be provided in demo mode")
            content_image_pil = None
        shading_image = shading_image if shading_image else Image.new('RGB', args.content_image_size)
        background_image = background_image if background_image else Image.new('RGB', args.content_image_size)

    content_inference_transforms = transforms.Compose(
        [
            transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    style_inference_transforms = transforms.Compose(
        [
            transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    shading_inference_transforms = transforms.Compose(
        [
            transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    background_inference_transforms = transforms.Compose(
        [
            transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]
    shading_image = shading_inference_transforms(shading_image)[None, :]
    background_image = background_inference_transforms(background_image)[None, :]

    return content_image, style_image, shading_image, background_image, content_image_pil

def load_fontdiffuser_pipeline(args):
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    shading_encoder = build_shading_encoder(args=args)
    shading_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/shading_encoder.pth"))
    background_encoder = build_background_encoder(args=args)
    background_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/background_encoder.pth"))
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        shading_encoder=shading_encoder,
        background_encoder=background_encoder,
    )
    model.to(args.device)
    logger.info("Loaded model state_dict successfully")

    train_scheduler = build_ddpm_scheduler(args=args)
    logger.info("Loaded training DDPM scheduler successfully")

    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
        target_dir=args.target_dir,
        content_dir=args.content_dir,
        style_dir=args.style_dir,
    )
    logger.info("Loaded DPM-Solver pipeline successfully")

    return pipe

def sampling(args, pipe, content_image=None, style_image=None, shading_image=None, background_image=None):
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)

    content_image, style_image, shading_image, background_image, content_image_pil = image_process(
        args=args,
        content_image=content_image,
        style_image=style_image,
        shading_image=shading_image,
        background_image=background_image,
    )
    if content_image is None:
        logger.error("Failed to process input images")
        return None

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        shading_image = shading_image.to(args.device)
        background_image = background_image.to(args.device)
        logger.info("Sampling with DPM-Solver++...")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
            shading_images=shading_image,
            background_images=background_image,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn,
        )
        end = time.time()

        if args.save_image:
            logger.info("Saving images...")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=content_image_pil,
                    content_image_path=None,
                    style_image_path=args.style_image_path,
                    shading_image_path=args.shading_image_path,
                    background_image_path=args.background_image_path,
                    resolution=args.resolution,
                )
            else:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=None,
                    content_image_path=args.content_image_path,
                    style_image_path=args.style_image_path,
                    shading_image_path=args.shading_image_path,
                    background_image_path=args.background_image_path,
                    resolution=args.resolution,
                )
            logger.info(f"Finished sampling, time taken: {end - start}s")
        return images[0]

def load_controlnet_pipeline(args, config_path="lllyasviel/sd-controlnet-canny", ckpt_path="runwayml/stable-diffusion-v1-5"):
    from diffusers import ControlNetModel, AutoencoderKL
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    controlnet = ControlNetModel.from_pretrained(
        config_path, torch_dtype=torch.float16, cache_dir=f"{args.ckpt_dir}/controlnet"
    )
    logger.info("Loaded ControlNet model successfully")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        ckpt_path, controlnet=controlnet, torch_dtype=torch.float16, cache_dir=f"{args.ckpt_dir}/controlnet_pipeline"
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    logger.info("Loaded ControlNet pipeline successfully")
    return pipe

def controlnet(text_prompt, pil_image, pipe):
    image = np.array(pil_image)
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(
        text_prompt,
        num_inference_steps=50,
        generator=generator,
        image=canny_image,
        output_type='pil',
    ).images[0]
    return image

def load_instructpix2pix_pipeline(args, ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe

def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=text_prompt,
        image=image,
        generator=generator,
        num_inference_steps=20,
        image_guidance_scale=1.1,
    ).images[0]
    return image

if __name__ == "__main__":
    args = arg_parse()
    pipe = load_fontdiffuser_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)