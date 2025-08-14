import os
import shutil
import random
from pathlib import Path

def reorganize_dataset():
    """Reorganize the dataset into the correct structure for training"""
    
    # Source directories
    source_dir = Path('brain_tumor_dataset/train/TEST  MRI')
    yes_source = source_dir / 'test yes'
    
    # Target directories
    train_dir = Path('brain_tumor_dataset/train')
    validation_dir = Path('brain_tumor_dataset/validation')
    
    # Create target directories
    for split_dir in [train_dir, validation_dir]:
        for class_dir in ['yes', 'no']:
            (split_dir / class_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    yes_images = list(yes_source.glob('*.jpg')) + list(yes_source.glob('*.jpeg')) + list(yes_source.glob('*.png'))
    no_images = list(source_dir.glob('no*.jpg')) + list(source_dir.glob('no*.jpeg')) + list(source_dir.glob('no*.png')) + \
                list(source_dir.glob('N*.jpg')) + list(source_dir.glob('N*.JPG')) + list(source_dir.glob('N*.jpeg'))
    
    print(f"Found {len(yes_images)} tumor images and {len(no_images)} non-tumor images")
    
    # Split into train/validation (80/20)
    random.seed(42)  # For reproducible splits
    
    # Tumor images
    random.shuffle(yes_images)
    split_idx = int(len(yes_images) * 0.8)
    train_yes = yes_images[:split_idx]
    val_yes = yes_images[split_idx:]
    
    # Non-tumor images
    random.shuffle(no_images)
    split_idx = int(len(no_images) * 0.8)
    train_no = no_images[:split_idx]
    val_no = no_images[split_idx:]
    
    print(f"Training: {len(train_yes)} tumor, {len(train_no)} non-tumor")
    print(f"Validation: {len(val_yes)} tumor, {len(val_no)} non-tumor")
    
    # Copy files to train directory
    for img_path in train_yes:
        shutil.copy2(img_path, train_dir / 'yes' / img_path.name)
    
    for img_path in train_no:
        shutil.copy2(img_path, train_dir / 'no' / img_path.name)
    
    # Copy files to validation directory
    for img_path in val_yes:
        shutil.copy2(img_path, validation_dir / 'yes' / img_path.name)
    
    for img_path in val_no:
        shutil.copy2(img_path, validation_dir / 'no' / img_path.name)
    
    print("Dataset reorganization completed!")

if __name__ == "__main__":
    reorganize_dataset()
