import os
import random
import logging
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.mixture import GaussianMixture

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_nonorm_transform(resolution):
    nonorm_transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    return nonorm_transform

class FontDataset(Dataset):
    """Dataset for font generation with shading and background support, adapted for TE141K structure"""
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase  # 'train' or 'val'
        self.scr = scr
        self.resolution = args.resolution
        self.target_dir = args.target_dir  # 'E'
        self.content_dir = args.content_dir  # 'C'
        self.style_dir = args.style_dir  # 'S'
        if self.scr:
            self.num_neg = args.num_neg
        
        # Initialize data structures
        self.target_images = []
        self.content_images = []
        self.style_images = []
        self.style_to_images = {}
        # Placeholder for shading/background (computed on-the-fly)
        self.shading_images = []
        self.background_images = []
        
        # Get data paths
        self.get_path()
        
        # Transforms
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(self.resolution)

    def segment_shading_background(self, style_image):
        """
        Segment a style image into shading and background components using GMM and Sobel edge detection.
        
        Args:
            style_image: PIL Image (RGB, ~256x256).
        
        Returns:
            shading_image: PIL Image (RGB, 128x128) with texture effects.
            background_image: PIL Image (RGB, 128x128) with background.
        """
        # Convert PIL to OpenCV
        img_np = np.array(style_image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Resize to 128x128 (as per args.resolution)
        img_np = cv2.resize(img_np, (128, 128), interpolation=cv2.INTER_AREA)
        
        # GMM for background
        pixel_vals = img_np.reshape((-1, 3)).astype(np.float32)
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(pixel_vals)
        labels = gmm.predict(pixel_vals)
        centers = gmm.means_.astype(np.uint8)
        segmented_img = centers[labels].reshape(img_np.shape)
        
        # Identify background cluster
        labels_2d = labels.reshape(img_np.shape[:2])
        variances = [np.var(segmented_img[labels_2d == i]) for i in range(3)]
        background_label = np.argmin(variances)
        background_mask = (labels_2d == background_label).astype(np.uint8) * 255
        
        # Inpaint character regions for background
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, char_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        inpaint_mask = cv2.bitwise_not(background_mask) | char_mask
        background_img = cv2.inpaint(img_np, inpaint_mask, 3, cv2.INPAINT_TELEA)
        
        # Shading: Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        _, shading = cv2.threshold(edge_magnitude, 50, 255, cv2.THRESH_BINARY)
        shading = shading.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        shading = cv2.dilate(shading, kernel, iterations=1)
        shading = cv2.bitwise_not(shading)
        shading_img = cv2.cvtColor(shading, cv2.COLOR_GRAY2BGR)
        
        # Mask shading
        shading_img = cv2.bitwise_and(shading_img, shading_img, mask=cv2.bitwise_not(background_mask))
        
        # Convert to PIL
        shading_image = Image.fromarray(cv2.cvtColor(shading_img, cv2.COLOR_BGR2RGB))
        background_image = Image.fromarray(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB))
        
        return shading_image, background_image

    def get_path(self):
        """Parse TE141K structure: E, C, S directories with style/train/val subdirs"""
        target_root = os.path.join(self.root, self.target_dir)
        content_root = os.path.join(self.root, self.content_dir)
        style_root = os.path.join(self.root, self.style_dir)

        # Validate and debug target_root
        logger.info(f"Checking target_root: {target_root}")
        if not os.path.exists(target_root):
            raise FileNotFoundError(f"Target directory {target_root} does not exist. Ensure data_root is correct (e.g., /root/sensei/te141k).")
        target_contents = os.listdir(target_root)
        logger.info(f"Contents of {target_root}: {target_contents}")

        # Collect target images from E
        for style in target_contents:
            style_path = os.path.join(target_root, style, self.phase)
            if not os.path.isdir(style_path):
                continue
            logger.info(f"Checking style_path: {style_path}")
            style_contents = os.listdir(style_path)
            logger.info(f"Contents of {style_path}: {style_contents}")
            images_related_style = []
            for img in sorted(style_contents):
                if img.endswith('.png') and not img.startswith('shading') and not img.startswith('background'):
                    img_path = os.path.join(style_path, img)
                    self.target_images.append(img_path)
                    images_related_style.append(img_path)
                    self.shading_images.append(None)  # Computed in __getitem__
                    self.background_images.append(None)
            self.style_to_images[style] = images_related_style

        # Validate and debug content_root
        logger.info(f"Checking content_root: {content_root}")
        if not os.path.exists(content_root):
            raise FileNotFoundError(f"Content directory {content_root} does not exist.")
        content_contents = os.listdir(content_root)
        logger.info(f"Contents of {content_root}: {content_contents}")

        # Collect content images from C
        for folder in content_contents:
            folder_path = os.path.join(content_root, folder, self.phase)
            if not os.path.isdir(folder_path):
                continue
            logger.info(f"Checking folder_path: {folder_path}")
            folder_contents = os.listdir(folder_path)
            logger.info(f"Contents of {folder_path}: {folder_contents}")
            for img in sorted(folder_contents):
                if img.endswith('.png'):
                    img_path = os.path.join(folder_path, img)
                    self.content_images.append(img_path)

        # Validate and debug style_root
        logger.info(f"Checking style_root: {style_root}")
        if not os.path.exists(style_root):
            raise FileNotFoundError(f"Style directory {style_root} does not exist.")
        style_contents = os.listdir(style_root)
        logger.info(f"Contents of {style_root}: {style_contents}")

        # Collect style images from S
        for style in style_contents:
            style_path = os.path.join(style_root, style, self.phase)
            if not os.path.isdir(style_path):
                continue
            logger.info(f"Checking style_path: {style_path}")
            style_path_contents = os.listdir(style_path)
            logger.info(f"Contents of {style_path}: {style_path_contents}")
            for img in sorted(style_path_contents):
                if img.endswith('.png'):
                    img_path = os.path.join(style_path, img)
                    self.style_images.append(img_path)

    def char_to_index(self, char):
        """Map character to index (placeholder; replace with metadata)"""
        cjk_start = 0x4E00
        index = ord(char) - cjk_start
        if index < 0 or index > 10000:
            logger.warning(f"Character {char} outside expected range, using index 0")
            index = 0
        return str(index)

    def index_to_char(self, index):
        """Map index to character (placeholder; replace with metadata)"""
        try:
            cjk_start = 0x4E00
            char = chr(int(index) + cjk_start)
            return char
        except ValueError:
            logger.warning(f"Invalid index {index}, using default character 一")
            return '一'

    def __getitem__(self, index):
        """Return a sample with content, style, target, shading, and background images"""
        target_image_path = self.target_images[index]
        target_image_name = os.path.basename(target_image_path)  # e.g., '0.png'
        
        # Extract style and content
        style = target_image_path.split(os.sep)[-3]  # e.g., 'Style1'
        content_idx = os.path.splitext(target_image_name)[0]  # e.g., '0'
        content_char = self.index_to_char(content_idx)
        
        # Read content image (from C, match index across randomized folders)
        content_image_path = None
        for folder in os.listdir(os.path.join(self.root, self.content_dir)):
            candidate_path = os.path.join(self.root, self.content_dir, folder, self.phase, f"{content_idx}.png")
            if os.path.exists(candidate_path):
                content_image_path = candidate_path
                break
        if not content_image_path:
            logger.warning(f"Content image missing for index {content_idx}, using blank")
            content_image = Image.new('RGB', (self.resolution, self.resolution))
        else:
            content_image = Image.open(content_image_path).convert('RGB')

        # Random sample for style image (from S or same style in E)
        images_related_style = self.style_to_images[style].copy()
        if target_image_path in images_related_style:
            images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style) if images_related_style else \
                          os.path.join(self.root, self.style_dir, style, self.phase, f"{content_idx}.png")
        if not os.path.exists(style_image_path):
            logger.warning(f"Style image missing: {style_image_path}, using blank")
            style_image = Image.new('RGB', (self.resolution, self.resolution))
        else:
            style_image = Image.open(style_image_path).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        # Compute shading and background on-the-fly
        shading_image, background_image = self.segment_shading_background(target_image)

        # Apply transforms
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
            shading_image = self.transforms[2](shading_image)
            background_image = self.transforms[2](background_image)
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "shading_image": shading_image,
            "background_image": background_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            "content_char": content_char,
        }
        
        if self.scr:
            # Get negative images from different styles for the same content
            style_list = list(self.style_to_images.keys())
            if style in style_list:
                style_list.remove(style)
            choose_neg_names = []
            for _ in range(self.num_neg):
                if not style_list:
                    break
                choose_style = random.choice(style_list)
                style_list.remove(choose_style)
                choose_neg_name = os.path.join(self.root, self.target_dir, choose_style, self.phase, f"{content_idx}.png")
                if os.path.exists(choose_neg_name):
                    choose_neg_names.append(choose_neg_name)

            # Load negative images
            neg_images = []
            for neg_name in choose_neg_names:
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                neg_images.append(neg_image[None, :, :, :])
            if neg_images:
                sample["neg_images"] = torch.cat(neg_images, dim=0)
            else:
                sample["neg_images"] = torch.zeros((self.num_neg, 3, self.resolution, self.resolution))

        return sample

    def __len__(self):
        return len(self.target_images)