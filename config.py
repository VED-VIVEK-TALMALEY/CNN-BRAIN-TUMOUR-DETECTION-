"""
Configuration file for Brain Tumor Detection Project
Centralizes all project paths and settings
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Source code directories
SRC_DIR = PROJECT_ROOT / "src"
MODELS_SRC_DIR = SRC_DIR / "models"
UTILS_SRC_DIR = SRC_DIR / "utils"
WEB_SRC_DIR = SRC_DIR / "web"
TRAINING_SRC_DIR = SRC_DIR / "training"

# Static and template directories
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Model file paths
MODEL_FILE = MODELS_DIR / "brain_tumor_cnn_model.h5"
MODEL_CONFIG_FILE = MODELS_DIR / "model_config.json"

# Dataset paths
DATASET_DIR = PROCESSED_DATA_DIR / "brain_tumor_dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TRAIN_YES_DIR = TRAIN_DIR / "yes"
TRAIN_NO_DIR = TRAIN_DIR / "no"
VALIDATION_YES_DIR = VALIDATION_DIR / "yes"
VALIDATION_NO_DIR = VALIDATION_DIR / "no"

# Web application settings
WEB_CONFIG = {
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'DEBUG': True,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'UPLOAD_FOLDER': str(UPLOADS_DIR),
    'SECRET_KEY': 'your-secret-key-here-change-in-production'
}

# Model training settings
TRAINING_CONFIG = {
    'INPUT_SHAPE': (150, 150, 3),
    'BATCH_SIZE': 32,
    'EPOCHS': 30,
    'LEARNING_RATE': 0.001,
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_SEED': 42
}

# Image processing settings
IMAGE_CONFIG = {
    'TARGET_SIZE': (150, 150),
    'SUPPORTED_FORMATS': {'.jpg', '.jpeg', '.png', '.bmp'},
    'NORMALIZATION': True,
    'AUGMENTATION': True
}

# Create necessary directories
def create_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        UPLOADS_DIR,
        TRAIN_YES_DIR,
        TRAIN_NO_DIR,
        VALIDATION_YES_DIR,
        VALIDATION_NO_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Get relative paths for web templates
def get_web_paths():
    """Get paths relative to project root for web templates"""
    return {
        'static_url': '/static',
        'css_url': '/static/css',
        'js_url': '/static/js',
        'images_url': '/static/images'
    }

# Validate configuration
def validate_config():
    """Validate that all required directories and files exist"""
    required_dirs = [
        SRC_DIR,
        MODELS_SRC_DIR,
        UTILS_SRC_DIR,
        WEB_SRC_DIR,
        TRAINING_SRC_DIR,
        STATIC_DIR,
        TEMPLATES_DIR
    ]
    
    missing_dirs = [str(d) for d in required_dirs if not d.exists()]
    
    if missing_dirs:
        raise FileNotFoundError(f"Missing required directories: {', '.join(missing_dirs)}")
    
    return True

if __name__ == "__main__":
    # Create directories when run directly
    create_directories()
    print("Project directories created successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
