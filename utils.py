import os
import cv2
import yaml
import copy
import pygame
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont

import torch
import torchvision.transforms as transforms


def save_args_to_yaml(args, output_file):
    """Save command-line arguments to a YAML file."""
    args_dict = vars(args)
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def save_single_image(save_dir, image):
    """Save a single image to the specified directory."""
    save_path = f"{save_dir}/out_single.png"
    image.save(save_path)


def save_image_with_content_style(
    save_dir, 
    image, 
    content_image_pil, 
    content_image_path, 
    style_image_path, 
    shading_image_path, 
    background_image_path, 
    resolution
):
    """Save a composite image with content, style, shading, background, and output images side by side."""
    new_image = Image.new('RGB', (resolution * 5, resolution))
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    shading_image = Image.open(shading_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    background_image = Image.open(background_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(shading_image, (resolution * 2, 0))
    new_image.paste(background_image, (resolution * 3, 0))
    new_image.paste(image, (resolution * 4, 0))

    save_path = f"{save_dir}/out_with_cs_sb.jpg"
    new_image.save(save_path)


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """
    Compute the predicted original sample (x_0) from noise prediction using the scheduler.
    
    Args:
        scheduler: Diffusion scheduler (e.g., DDPMScheduler).
        noise_pred: Predicted noise tensor of shape [batch_size, channels, height, width].
        x_t: Noisy input tensor at timestep t, same shape as noise_pred.
        timesteps: Timesteps tensor of shape [batch_size].
    
    Returns:
        pred_original_sample: Predicted original sample tensor of shape [batch_size, channels, height, width].
    """
    batch_size = noise_pred.shape[0]
    pred_original_sample = torch.zeros_like(noise_pred)

    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]  # Add batch dimension
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]  # Add batch dimension

        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        pred_original_sample[i] = pred_original_sample_i[0]  # Remove batch dimension and assign

    return pred_original_sample


def reNormalize_img(image):
    """
    Re-normalize image from [-1, 1] to [0, 1] for visualization.
    
    Args:
        image: Tensor of shape [batch_size, channels, height, width] in range [-1, 1].
    
    Returns:
        Tensor of same shape in range [0, 1].
    """
    return (image + 1) / 2


def normalize_mean_std(image):
    """
    Normalize image by subtracting mean and dividing by standard deviation.
    
    Args:
        image: Tensor of shape [batch_size, channels, height, width].
    
    Returns:
        Normalized tensor of same shape.
    """
    mean = image.mean(dim=(1, 2, 3), keepdim=True)
    std = image.std(dim=(1, 2, 3), keepdim=True)
    return (image - mean) / (std + 1e-6)


def is_char_in_font(font_path, char):
    """
    Check if a character is supported by the given TTF font.
    
    Args:
        font_path: Path to the TTF font file.
        char: Single character to check.
    
    Returns:
        bool: True if character is in font, False otherwise.
    """
    font = TTFont(font_path)
    for table in font['cmap'].tables:
        if ord(char) in table.cmap:
            return True
    return False


def load_ttf(ttf_path):
    """
    Load a TTF font using pygame.
    
    Args:
        ttf_path: Path to the TTF font file.
    
    Returns:
        pygame.font.Font object.
    """
    pygame.font.init()
    return pygame.font.Font(ttf_path, 128)


def ttf2im(font, char):
    """
    Render a character from a TTF font to a PIL image.
    
    Args:
        font: pygame.font.Font object.
        char: Single character to render.
    
    Returns:
        PIL.Image: Rendered character image.
    """
    text_surface = font.render(char, True, (255, 255, 255), (0, 0, 0))
    image = pygame.surfarray.array3d(text_surface)
    image = np.transpose(image, (1, 0, 2))
    return Image.fromarray(image)


def segment_shading_background(image_path, shading_output_path=None, background_output_path=None):
    """
    Segment an input image into shading and background components using OpenCV.
    
    Args:
        image_path: Path to the input image (RGB).
        shading_output_path: Optional path to save the shading image.
        background_output_path: Optional path to save the background image.
    
    Returns:
        tuple: (shading_image, background_image) as PIL.Image objects.
    """
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shading: Extract high-frequency textures (edges, gradients)
    # Apply Sobel edge detection for texture details
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Enhance texture with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    shading = cv2.dilate(sobel, kernel, iterations=1)
    shading = cv2.erode(shading, kernel, iterations=1)

    # Convert shading to 3-channel for compatibility
    shading = cv2.cvtColor(shading, cv2.COLOR_GRAY2RGB)

    # Background: Extract low-frequency color regions
    # Apply Gaussian blur to isolate low-frequency components
    blurred = cv2.GaussianBlur(img_rgb, (15, 15), 0)

    # Use k-means clustering to segment dominant background colors
    pixel_values = blurred.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    background = centers[labels.flatten()].reshape(img_rgb.shape)

    # Refine background with morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)

    # Convert to PIL images
    shading_pil = Image.fromarray(shading)
    background_pil = Image.fromarray(background)

    # Save images if paths are provided
    if shading_output_path:
        shading_pil.save(shading_output_path)
    if background_output_path:
        background_pil.save(background_output_path)

    return shading_pil, background_pil