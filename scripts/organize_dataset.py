#!/usr/bin/env python3
"""
Dataset Organization Script for Brain Tumor Detection Project
Organizes raw dataset into train/validation splits
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import random

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import (
    RAW_DATA_DIR, TRAIN_DIR, VALIDATION_DIR, 
    TRAIN_YES_DIR, TRAIN_NO_DIR, VALIDATION_YES_DIR, VALIDATION_NO_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def organize_dataset(train_split=0.8, random_seed=42):
    """
    Organize the dataset into train/validation splits
    
    Args:
        train_split (float): Proportion of data to use for training (default: 0.8)
        random_seed (int): Random seed for reproducibility (default: 42)
    """
    logger.info("🧠 Starting dataset organization...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Check if raw data exists
    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        logger.info("Please place your brain tumor dataset in the 'data/raw' directory")
        return False
    
    # Create necessary directories
    for directory in [TRAIN_YES_DIR, TRAIN_NO_DIR, VALIDATION_YES_DIR, VALIDATION_NO_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    # Look for dataset structure
    dataset_found = False
    
    # Check if dataset is already organized
    if TRAIN_DIR.exists() and any(TRAIN_YES_DIR.iterdir()) and any(TRAIN_NO_DIR.iterdir()):
        logger.info("Dataset appears to be already organized!")
        return True
    
    # Look for common dataset structures
    possible_paths = [
        RAW_DATA_DIR / "brain_tumor_dataset",
        RAW_DATA_DIR / "dataset",
        RAW_DATA_DIR / "data",
        RAW_DATA_DIR
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            # Look for yes/no directories
            yes_dirs = list(base_path.glob("**/yes"))
            no_dirs = list(base_path.glob("**/no"))
            
            if yes_dirs and no_dirs:
                logger.info(f"Found dataset structure in: {base_path}")
                dataset_found = True
                
                # Get all image files
                yes_files = []
                no_files = []
                
                for yes_dir in yes_dirs:
                    yes_files.extend([f for f in yes_dir.iterdir() if f.suffix.lower() in image_extensions])
                
                for no_dir in no_dirs:
                    no_files.extend([f for f in no_dir.iterdir() if f.suffix.lower() in image_extensions])
                
                logger.info(f"Found {len(yes_files)} positive samples and {len(no_files)} negative samples")
                
                # Split into train/validation
                random.shuffle(yes_files)
                random.shuffle(no_files)
                
                train_yes_count = int(len(yes_files) * train_split)
                train_no_count = int(len(no_files) * train_split)
                
                # Copy training files
                for i, file_path in enumerate(yes_files[:train_yes_count]):
                    dest_path = TRAIN_YES_DIR / f"yes_{i:04d}{file_path.suffix}"
                    shutil.copy2(file_path, dest_path)
                
                for i, file_path in enumerate(no_files[:train_no_count]):
                    dest_path = TRAIN_NO_DIR / f"no_{i:04d}{file_path.suffix}"
                    shutil.copy2(file_path, dest_path)
                
                # Copy validation files
                for i, file_path in enumerate(yes_files[train_yes_count:]):
                    dest_path = VALIDATION_YES_DIR / f"yes_{i:04d}{file_path.suffix}"
                    shutil.copy2(file_path, dest_path)
                
                for i, file_path in enumerate(no_files[train_no_count:]):
                    dest_path = VALIDATION_NO_DIR / f"no_{i:04d}{file_path.suffix}"
                    shutil.copy2(file_path, dest_path)
                
                logger.info(f"✅ Dataset organized successfully!")
                logger.info(f"   Training: {train_yes_count} positive, {train_no_count} negative")
                logger.info(f"   Validation: {len(yes_files) - train_yes_count} positive, {len(no_files) - train_no_count} negative")
                
                return True
    
    if not dataset_found:
        logger.error("Could not find organized dataset structure")
        logger.info("Expected structure:")
        logger.info("  data/raw/brain_tumor_dataset/")
        logger.info("  ├── train/")
        logger.info("  │   ├── yes/  (tumor images)")
        logger.info("  │   └── no/   (no tumor images)")
        logger.info("  └── validation/")
        logger.info("      ├── yes/  (tumor images)")
        logger.info("      └── no/   (no tumor images)")
        return False
    
    return True

def main():
    """Main function"""
    try:
        success = organize_dataset()
        if success:
            logger.info("🎉 Dataset organization completed successfully!")
        else:
            logger.error("❌ Dataset organization failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during dataset organization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
