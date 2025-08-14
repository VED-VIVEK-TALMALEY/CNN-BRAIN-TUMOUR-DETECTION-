"""
Image Processing Utilities for Brain Tumor Detection
"""

import numpy as np
from PIL import Image
import cv2
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import IMAGE_CONFIG

logger = logging.getLogger(__name__)

def load_image(image_path):
    """
    Load an image from file path
    
    Args:
        image_path: Path to the image file
    
    Returns:
        PIL Image object
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise

def preprocess_image(image_path, target_size=None):
    """
    Preprocess image for model prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (default: from config)
    
    Returns:
        Preprocessed image array ready for model input
    """
    if target_size is None:
        target_size = IMAGE_CONFIG['TARGET_SIZE']
    
    try:
        # Load image
        image = load_image(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values
        if IMAGE_CONFIG['NORMALIZATION']:
            image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise ValueError(f"Failed to process image: {str(e)}")

def preprocess_batch(image_paths, target_size=None):
    """
    Preprocess multiple images for batch prediction
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for the images
    
    Returns:
        Batch of preprocessed images
    """
    if target_size is None:
        target_size = IMAGE_CONFIG['TARGET_SIZE']
    
    processed_images = []
    
    for image_path in image_paths:
        try:
            processed_image = preprocess_image(image_path, target_size)
            processed_images.append(processed_image)
        except Exception as e:
            logger.warning(f"Skipping image {image_path}: {e}")
            continue
    
    if not processed_images:
        raise ValueError("No images were successfully processed")
    
    # Stack all images into a single batch
    batch = np.vstack(processed_images)
    return batch

def validate_image_format(image_path):
    """
    Validate if the image format is supported
    
    Args:
        image_path: Path to the image file
    
    Returns:
        True if format is supported, False otherwise
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    return file_ext in IMAGE_CONFIG['SUPPORTED_FORMATS']

def get_image_info(image_path):
    """
    Get basic information about an image
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with image information
    """
    try:
        image = load_image(image_path)
        
        info = {
            'path': image_path,
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'file_size': os.path.getsize(image_path),
            'supported': validate_image_format(image_path)
        }
        
        return info
    except Exception as e:
        logger.error(f"Error getting image info for {image_path}: {e}")
        return None

def apply_augmentation(image_array, augmentation_type='basic'):
    """
    Apply data augmentation to image array
    
    Args:
        image_array: Input image array
        augmentation_type: Type of augmentation to apply
    
    Returns:
        Augmented image array
    """
    if not IMAGE_CONFIG['AUGMENTATION']:
        return image_array
    
    try:
        if augmentation_type == 'basic':
            # Basic augmentation: random brightness and contrast
            augmented = image_array.copy()
            
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * brightness_factor, 0, 1)
            
            # Random contrast adjustment
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(augmented)
            augmented = np.clip((augmented - mean) * contrast_factor + mean, 0, 1)
            
            return augmented
        
        elif augmentation_type == 'advanced':
            # Advanced augmentation using OpenCV
            augmented = image_array.copy()
            
            # Convert to uint8 for OpenCV operations
            img_uint8 = (augmented[0] * 255).astype(np.uint8)
            
            # Random rotation
            angle = np.random.uniform(-30, 30)
            height, width = img_uint8.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_uint8 = cv2.warpAffine(img_uint8, rotation_matrix, (width, height))
            
            # Random noise
            noise = np.random.normal(0, 0.05, img_uint8.shape).astype(np.uint8)
            img_uint8 = cv2.add(img_uint8, noise)
            
            # Convert back to float32 and normalize
            augmented[0] = img_uint8.astype(np.float32) / 255.0
            
            return augmented
        
        else:
            logger.warning(f"Unknown augmentation type: {augmentation_type}")
            return image_array
            
    except Exception as e:
        logger.error(f"Error applying augmentation: {e}")
        return image_array

def save_processed_image(image_array, output_path, format='PNG'):
    """
    Save a processed image array to file
    
    Args:
        image_array: Image array to save
        output_path: Output file path
        format: Image format to save as
    """
    try:
        # Remove batch dimension if present
        if len(image_array.shape) == 4:
            image_array = image_array[0]
        
        # Convert to uint8 if normalized
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        image.save(output_path, format=format)
        logger.info(f"Processed image saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving processed image: {e}")
        raise

def create_image_grid(image_arrays, output_path, grid_size=(3, 3)):
    """
    Create a grid of images for visualization
    
    Args:
        image_arrays: List of image arrays
        output_path: Output file path
        grid_size: Grid dimensions (rows, cols)
    """
    try:
        if len(image_arrays) == 0:
            raise ValueError("No images provided")
        
        # Limit to grid size
        max_images = grid_size[0] * grid_size[1]
        image_arrays = image_arrays[:max_images]
        
        # Create grid
        rows, cols = grid_size
        grid_height = rows * image_arrays[0].shape[0]
        grid_width = cols * image_arrays[0].shape[1]
        
        # Create empty grid
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Fill grid with images
        for idx, img_array in enumerate(image_arrays):
            if idx >= max_images:
                break
                
            row = idx // cols
            col = idx % cols
            
            # Convert to uint8 if normalized
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            
            # Remove batch dimension if present
            if len(img_array.shape) == 4:
                img_array = img_array[0]
            
            # Calculate position in grid
            y_start = row * img_array.shape[0]
            y_end = y_start + img_array.shape[0]
            x_start = col * img_array.shape[1]
            x_end = x_start + img_array.shape[1]
            
            # Place image in grid
            grid[y_start:y_end, x_start:x_end] = img_array
        
        # Save grid
        grid_image = Image.fromarray(grid)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid_image.save(output_path)
        logger.info(f"Image grid saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating image grid: {e}")
        raise

if __name__ == "__main__":
    # Test image processing functions
    print("Image processing utilities loaded successfully!")
    print(f"Supported formats: {IMAGE_CONFIG['SUPPORTED_FORMATS']}")
    print(f"Target size: {IMAGE_CONFIG['TARGET_SIZE']}")
    print(f"Normalization: {IMAGE_CONFIG['NORMALIZATION']}")
    print(f"Augmentation: {IMAGE_CONFIG['AUGMENTATION']}")
