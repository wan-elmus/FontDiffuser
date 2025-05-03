import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def segment_shading_background(image_path):
    """
    Segment a style image into shading and background components using OpenCV.
    This is a basic implementation; replace with a more robust method if needed.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Shading: Detect edges/high-frequency components
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    shading = cv2.dilate(edges, kernel, iterations=1)
    shading = cv2.cvtColor(shading, cv2.COLOR_GRAY2BGR)  # Convert to RGB
    
    # Background: Smooth image to capture low-frequency components
    background = cv2.GaussianBlur(img, (21, 21), 0)
    
    # Resize to 256x256 if needed
    shading = cv2.resize(shading, (256, 256), interpolation=cv2.INTER_AREA)
    background = cv2.resize(background, (256, 256), interpolation=cv2.INTER_AREA)
    
    return shading, background

def process_style_directory(style_dir, output_dir):
    """
    Process all images in a style directory, creating shading/ and background/ subdirs.
    """
    style_path = Path(style_dir)
    for phase in ["train", "val"]:
        phase_path = style_path / phase
        if not phase_path.exists():
            continue
            
        # Create shading and background directories
        shading_dir = phase_path / "shading"
        background_dir = phase_path / "background"
        shading_dir.mkdir(exist_ok=True)
        background_dir.mkdir(exist_ok=True)
        
        # Process each image
        image_files = list(phase_path.glob("*.png"))
        for img_path in tqdm(image_files, desc=f"Processing {style_path.name}/{phase}"):
            if img_path.name in ["shading", "background"]:
                continue
            try:
                shading_img, background_img = segment_shading_background(str(img_path))
                
                # Save images
                cv2.imwrite(str(shading_dir / img_path.name), shading_img)
                cv2.imwrite(str(background_dir / img_path.name), background_img)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess TE141K dataset for shading and background.")
    parser.add_argument("--data_dir", type=str, default="te141k", help="Path to TE141K dataset")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Dataset directory {data_dir} does not exist")
    
    # Process each style directory under te141k/E/
    e_dir = data_dir / "E"
    if not e_dir.exists():
        raise ValueError(f"Style directory {e_dir} does not exist")
    
    for style_dir in e_dir.iterdir():
        if style_dir.is_dir():
            print(f"Processing style: {style_dir.name}")
            process_style_directory(style_dir, data_dir)
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()