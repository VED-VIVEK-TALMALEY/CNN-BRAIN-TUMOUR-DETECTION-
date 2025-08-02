import os
import logging
from cnn_model import train_model
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    train_dir = 'brain_tumor_dataset/train'
    validation_dir = 'brain_tumor_dataset/validation'
    
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        raise ValueError("Training or validation directory does not exist")
    
    if not os.listdir(train_dir) or not os.listdir(validation_dir):
        raise ValueError("Training or validation directory is empty")
    
    if not any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(train_dir)):
        raise ValueError("No image files found in training directory")
    
    if not any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(validation_dir)):
        raise ValueError("No image files found in validation directory")
    
    batch_size = 32
    epochs = 30
    
    model, history = train_model(
        train_dir=train_dir,
        validation_dir=validation_dir,
        batch_size=batch_size,
        epochs=epochs
    )
    
    model.save('brain_tumor_cnn_model.h5')
    
    print("\nTraining History:")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == '__main__':
    main() 