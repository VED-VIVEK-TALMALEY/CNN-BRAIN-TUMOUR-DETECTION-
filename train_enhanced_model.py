import os
import sys
import numpy as np
from enhanced_cnn_model import (
    train_enhanced_model, 
    evaluate_model_performance,
    create_advanced_data_generators
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_balanced_dataset(source_dir, target_dir, max_samples_per_class=None):
    """
    Create a balanced dataset with equal representation of both classes
    """
    logger.info("Creating balanced dataset...")
    
    # Create target directories
    os.makedirs(os.path.join(target_dir, 'train', 'yes'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train', 'no'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'validation', 'yes'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'validation', 'no'), exist_ok=True)
    
    # Get all image files
    yes_images = []
    no_images = []
    
    # Collect tumor images
    yes_source = os.path.join(source_dir, 'train', 'TEST  MRI', 'test yes')
    if os.path.exists(yes_source):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            yes_images.extend(list(Path(yes_source).glob(ext)))
    
    # Collect non-tumor images
    no_source = os.path.join(source_dir, 'train', 'TEST  MRI')
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        no_images.extend([f for f in Path(no_source).glob(ext) 
                         if f.name.lower().startswith('no') or f.name.lower().startswith('n')])
    
    logger.info(f"Found {len(yes_images)} tumor images and {len(no_images)} non-tumor images")
    
    # Balance the dataset
    min_samples = min(len(yes_images), len(no_images))
    if max_samples_per_class:
        min_samples = min(min_samples, max_samples_per_class)
    
    # Randomly sample equal numbers from each class
    np.random.seed(42)
    yes_selected = np.random.choice(yes_images, min_samples, replace=False)
    no_selected = np.random.choice(no_images, min_samples, replace=False)
    
    # Split into train/validation (80/20)
    train_split = int(0.8 * min_samples)
    
    # Copy tumor images
    for i, img_path in enumerate(yes_selected):
        if i < train_split:
            dest = os.path.join(target_dir, 'train', 'yes', img_path.name)
        else:
            dest = os.path.join(target_dir, 'validation', 'yes', img_path.name)
        shutil.copy2(img_path, dest)
    
    # Copy non-tumor images
    for i, img_path in enumerate(no_selected):
        if i < train_split:
            dest = os.path.join(target_dir, 'train', 'no', img_path.name)
        else:
            dest = os.path.join(target_dir, 'validation', 'no', img_path.name)
        shutil.copy2(img_path, dest)
    
    logger.info(f"Created balanced dataset with {min_samples} samples per class")
    logger.info(f"Training: {train_split} samples per class")
    logger.info(f"Validation: {min_samples - train_split} samples per class")
    
    return min_samples

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history with multiple metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Enhanced Brain Tumor Detection Model', fontsize=16)
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Training Loss')
    axes[0, 1].plot(history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history:
        axes[1, 0].plot(history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Precision and Recall (if available)
    if 'precision' in history and 'recall' in history:
        axes[1, 1].plot(history['precision'], label='Precision')
        axes[1, 1].plot(history['recall'], label='Recall')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function with enhanced techniques
    """
    try:
        # Configuration
        source_dataset = 'brain_tumor_dataset'
        enhanced_dataset = 'enhanced_brain_tumor_dataset'
        model_type = 'transfer'  # 'transfer' or 'custom'
        input_size = 224  # Increased from 150x150
        batch_size = 16   # Reduced for better gradient updates
        epochs = 100      # Increased epochs
        
        logger.info("Starting enhanced brain tumor detection model training...")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Input size: {input_size}x{input_size}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Epochs: {epochs}")
        
        # Create balanced dataset
        if not os.path.exists(enhanced_dataset):
            create_balanced_dataset(source_dataset, enhanced_dataset, max_samples_per_class=200)
        else:
            logger.info("Enhanced dataset already exists, skipping creation...")
        
        # Define paths
        train_dir = os.path.join(enhanced_dataset, 'train')
        validation_dir = os.path.join(enhanced_dataset, 'validation')
        
        # Verify dataset structure
        if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
            logger.error("Dataset directories not found!")
            return
        
        # Count samples
        train_yes = len([f for f in os.listdir(os.path.join(train_dir, 'yes')) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_no = len([f for f in os.listdir(os.path.join(train_dir, 'no')) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_yes = len([f for f in os.listdir(os.path.join(validation_dir, 'yes')) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_no = len([f for f in os.listdir(os.path.join(validation_dir, 'no')) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        logger.info(f"Training samples: {train_yes} tumor, {train_no} non-tumor")
        logger.info(f"Validation samples: {val_yes} tumor, {val_no} non-tumor")
        
        # Train the enhanced model
        logger.info("Training enhanced model...")
        model, history = train_enhanced_model(
            train_dir=train_dir,
            validation_dir=validation_dir,
            model_type=model_type,
            batch_size=batch_size,
            epochs=epochs,
            input_size=input_size
        )
        
        # Evaluate model performance
        logger.info("Evaluating model performance...")
        train_datagen, validation_datagen = create_advanced_data_generators()
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        evaluation = evaluate_model_performance(model, validation_generator)
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history)
        
        # Save final model
        final_model_path = f'final_enhanced_brain_tumor_model_{model_type}.h5'
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Print final results
        final_accuracy = history['val_accuracy'][-1]
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Final validation accuracy: {final_accuracy:.4f}")
        
        if final_accuracy > 0.90:
            logger.info("🎉 TARGET ACHIEVED: Model accuracy > 90%!")
        elif final_accuracy > 0.85:
            logger.info("✅ GOOD: Model accuracy > 85%")
        else:
            logger.info("⚠️  Model accuracy below 85%. Consider:")
            logger.info("   - Increasing dataset size")
            logger.info("   - Using more data augmentation")
            logger.info("   - Trying different model architectures")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
