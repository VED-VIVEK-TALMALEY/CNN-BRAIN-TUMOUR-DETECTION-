"""
Training Script for Brain Tumor Detection CNN Model
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import (
    TRAINING_CONFIG, 
    DATASET_DIR, 
    TRAIN_DIR, 
    VALIDATION_DIR, 
    MODEL_FILE,
    create_directories
)
from src.models.cnn_model import train_model, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_dataset():
    """Validate that the dataset is properly organized"""
    required_dirs = [
        TRAIN_YES_DIR,
        TRAIN_NO_DIR,
        VALIDATION_YES_DIR,
        VALIDATION_NO_DIR
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not directory.exists():
            missing_dirs.append(str(directory))
    
    if missing_dirs:
        logger.error(f"Missing required directories: {', '.join(missing_dirs)}")
        return False
    
    # Check for images in each directory
    train_yes_count = len(list(TRAIN_YES_DIR.glob('*')))
    train_no_count = len(list(TRAIN_NO_DIR.glob('*')))
    val_yes_count = len(list(VALIDATION_YES_DIR.glob('*')))
    val_no_count = len(list(VALIDATION_NO_DIR.glob('*')))
    
    logger.info(f"Dataset validation results:")
    logger.info(f"  Training - Tumor images: {train_yes_count}")
    logger.info(f"  Training - No tumor images: {train_no_count}")
    logger.info(f"  Validation - Tumor images: {val_yes_count}")
    logger.info(f"  Validation - No tumor images: {val_no_count}")
    
    # Check minimum requirements
    if train_yes_count < 10 or train_no_count < 10:
        logger.error("Insufficient training data. Need at least 10 images per class.")
        return False
    
    if val_yes_count < 5 or val_no_count < 5:
        logger.error("Insufficient validation data. Need at least 5 images per class.")
        return False
    
    return True

def count_images(directory):
    """Count image files in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    count = 0
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            count += 1
    
    return count

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Detection CNN Model')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['EPOCHS'],
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['BATCH_SIZE'],
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=TRAINING_CONFIG['LEARNING_RATE'],
                       help='Learning rate')
    parser.add_argument('--model-path', type=str, default=str(MODEL_FILE),
                       help='Path to save the trained model')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset without training')
    
    args = parser.parse_args()
    
    logger.info("🧠 Brain Tumor Detection Model Training")
    logger.info("=" * 50)
    
    try:
        # Create necessary directories
        create_directories()
        logger.info("Project directories created/verified")
        
        # Validate dataset
        if not validate_dataset():
            logger.error("Dataset validation failed. Please organize your dataset properly.")
            logger.info("Use the organize_dataset.py script to organize your data.")
            return
        
        if args.validate_only:
            logger.info("Dataset validation completed successfully!")
            return
        
        # Training parameters
        training_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }
        
        logger.info(f"Training parameters:")
        logger.info(f"  Epochs: {training_params['epochs']}")
        logger.info(f"  Batch size: {training_params['batch_size']}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Model save path: {args.model_path}")
        
        # Start training
        logger.info("Starting model training...")
        model, history = train_model(
            train_dir=str(TRAIN_DIR),
            validation_dir=str(VALIDATION_DIR),
            model_save_path=args.model_path,
            **training_params
        )
        
        # Training summary
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info(f"Final training accuracy: {final_accuracy:.4f}")
        logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")
        logger.info(f"Final training loss: {final_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        logger.info(f"Model saved to: {args.model_path}")
        
        # Optional: Evaluate on validation set
        logger.info("\nEvaluating model on validation set...")
        metrics = evaluate_model(model, str(VALIDATION_DIR))
        
        logger.info("\nModel evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\n🎉 Training completed! You can now run the web application.")
        logger.info("Run: python src/web/app.py")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
