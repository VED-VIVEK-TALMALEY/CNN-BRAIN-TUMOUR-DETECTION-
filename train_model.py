import os
import sys
from cnn_model import create_model, train_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    try:
        # Define dataset paths
        train_dir = 'brain_tumor_dataset/train'
        validation_dir = 'brain_tumor_dataset/validation'
        
        # Check if dataset directories exist
        if not os.path.exists(train_dir):
            logger.error(f"Training directory not found: {train_dir}")
            logger.info("Please ensure your dataset is organized as follows:")
            logger.info("brain_tumor_dataset/")
            logger.info("├── train/")
            logger.info("│   ├── yes/ (tumor images)")
            logger.info("│   └── no/ (no tumor images)")
            logger.info("└── validation/")
            logger.info("    ├── yes/ (tumor images)")
            logger.info("    └── no/ (no tumor images)")
            return
        
        if not os.path.exists(validation_dir):
            logger.warning(f"Validation directory not found: {validation_dir}")
            logger.info("Creating validation directory...")
            os.makedirs(validation_dir, exist_ok=True)
            
            # Create yes/no subdirectories
            os.makedirs(os.path.join(validation_dir, 'yes'), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, 'no'), exist_ok=True)
        
        # Check if we have training data
        train_yes_dir = os.path.join(train_dir, 'yes')
        train_no_dir = os.path.join(train_dir, 'no')
        
        if not os.path.exists(train_yes_dir) or not os.path.exists(train_no_dir):
            logger.error("Training data not properly organized")
            logger.info("Please organize your training data into 'yes' and 'no' subdirectories")
            return
        
        # Count training samples
        train_yes_count = len([f for f in os.listdir(train_yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_no_count = len([f for f in os.listdir(train_no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        logger.info(f"Found {train_yes_count} tumor images and {train_no_count} non-tumor images")
        
        if train_yes_count == 0 or train_no_count == 0:
            logger.error("Insufficient training data")
            return
        
        # Training parameters
        batch_size = 32
        epochs = 30
        
        logger.info("Starting model training...")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Epochs: {epochs}")
        
        # Train the model
        model, history = train_model(
            train_dir=train_dir,
            validation_dir=validation_dir,
            batch_size=batch_size,
            epochs=epochs
        )
        
        # Save the trained model
        model_path = 'brain_tumor_cnn_model.h5'
        model.save(model_path)
        logger.info(f"Model saved successfully to {model_path}")
        
        # Print training summary
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        
        logger.info(f"Training completed!")
        logger.info(f"Final training accuracy: {final_accuracy:.4f}")
        logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()