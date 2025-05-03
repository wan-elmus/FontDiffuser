import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Training config for FontDiffuser.")
    ################# Experience #################
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--experience_name", type=str, default="fontdiffuser_training")
    parser.add_argument("--data_root", type=str, default="/sensei/te141k", 
                        help="The font dataset root path (e.g., /sensei/te141k).")
    parser.add_argument("--target_dir", type=str, default="E", 
                        help="Directory for target images (style+content combinations).")
    parser.add_argument("--content_dir", type=str, default="C", 
                        help="Directory for content images (character glyphs).")
    parser.add_argument("--style_dir", type=str, default="S", 
                        help="Directory for style reference images.")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf",
                        help="Path to TTF font file for character rendering.")
    parser.add_argument("--output_dir", type=str, default="outputs/FontDiffuser", 
                        help="The output directory for model predictions and checkpoints.")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs", 
                        help="TensorBoard log directory.")

    # Model
    parser.add_argument("--resolution", type=int, default=128, 
                        help="The resolution for input images.")
    parser.add_argument("--unet_channels", type=tuple, default=(320, 640, 1280, 1280),
                        help="The channels of the UNet.")
    parser.add_argument("--style_image_size", type=int, default=128, help="The size of style images.")
    parser.add_argument("--content_image_size", type=int, default=128, help="The size of content images.")
    parser.add_argument("--content_encoder_downsample_size", type=int, default=8, 
                        help="The downsample size of the content encoder.")
    parser.add_argument("--channel_attn", type=bool, default=True, help="Whether to use channel attention.")
    parser.add_argument("--content_start_channel", type=int, default=64, 
                        help="The channels of the first layer output of content encoder.")
    parser.add_argument("--style_start_channel", type=int, default=80, 
                        help="The channels of the first layer output of style encoder.")
    parser.add_argument("--shading_start_channel", type=int, default=64, 
                        help="The channels of the first layer output of shading encoder.")
    parser.add_argument("--background_start_channel", type=int, default=48, 
                        help="The channels of the first layer output of background encoder.")
    
    # Training
    parser.add_argument("--phase_2", action="store_true", help="Training in phase 2 using SCR module.")
    parser.add_argument("--phase_1_ckpt_dir", type=str, default="phase_1_ckpt", help="The trained ckpt directory during phase 1.")
    ## SCR
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mode", type=str, default="refinement")
    parser.add_argument("--scr_image_size", type=int, default=128)
    parser.add_argument("--scr_ckpt_path", type=str, default="ckpt/scr_210000.pth")
    parser.add_argument("--num_neg", type=int, default=16, help="Number of negative samples.")
    parser.add_argument("--nce_layers", type=str, default='0,4,8,12,16')
    parser.add_argument("--sc_coefficient", type=float, default=0.1)
    ## train batch size
    parser.add_argument("--train_batch_size", type=int, default=4, 
                        help="Batch size (per device) for the training dataloader.")
    ## loss coefficient
    parser.add_argument("--perceptual_coefficient", type=float, default=0.1)
    parser.add_argument("--offset_coefficient", type=float, default=0.02)
    parser.add_argument("--shading_coefficient", type=float, default=0.1, 
                        help="Coefficient for shading texture loss.")
    parser.add_argument("--background_coefficient", type=float, default=0.1, 
                        help="Coefficient for background perceptual loss.")
    parser.add_argument("--use_lpips", type=bool, default=True, 
                        help="Use LPIPS instead of VGG for background perceptual loss.")
    ## step
    parser.add_argument("--max_train_steps", type=int, default=440000, 
                        help="Total number of training steps to perform.")
    parser.add_argument("--ckpt_interval", type=int, default=40000, help="The step begin to validate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate.")
    parser.add_argument("--log_interval", type=int, default=100, help="The log interval of training.")
    ## learning rate
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Initial learning rate to use.")
    parser.add_argument("--scale_lr", type=bool, default=True, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size. Set to True or False.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                        help="The scheduler type to use: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'")
    parser.add_argument("--lr_warmup_steps", type=int, default=5000, 
                        help="Number of steps for the warmup in the lr scheduler.")
    ## classifier-free
    parser.add_argument("--drop_prob", type=float, default=0.1, help="The uncondition training drop out probability.")
    ## scheduler
    parser.add_argument("--beta_scheduler", type=str, default="scaled_linear", help="The beta scheduler for DDPM.")
    ## optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--mixed_precision", type=str, default="fp16", 
                        help="Whether to use mixed precision: 'no', 'fp16', 'bf16'.")
    
    # Sampling
    parser.add_argument("--algorithm_type", type=str, default="dpmsolver++", help="Algorithm for sampling.")
    parser.add_argument("--guidance_type", type=str, default="classifier-free", help="Guidance type of sampling.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale of the classifier-free mode.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Sampling step.")
    parser.add_argument("--model_type", type=str, default="noise", help="model_type for sampling.")
    parser.add_argument("--order", type=int, default=2, help="The order of the dpmsolver.")
    parser.add_argument("--skip_type", type=str, default="time_uniform", help="Skip type of dpmsolver.")
    parser.add_argument("--method", type=str, default="multistep", help="Multistep of dpmsolver.")
    parser.add_argument("--correcting_x0_fn", type=str, default=None, help="correcting_x0_fn of dpmsolver.")
    parser.add_argument("--t_start", type=int, default=1000, help="t_start of dpmsolver.")
    parser.add_argument("--t_end", type=int, default=1, help="t_end of dpmsolver.")
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    return parser