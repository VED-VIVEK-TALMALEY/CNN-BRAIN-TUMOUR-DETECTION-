import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Add, Input
)
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_model(input_shape=(224, 224, 3)):
    """
    Create an enhanced CNN model with advanced techniques for >90% accuracy
    """
    # Use transfer learning with pre-trained ResNet50V2
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with optimized parameters
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    return model, base_model

def create_custom_cnn_model(input_shape=(224, 224, 3)):
    """
    Create a custom CNN model with advanced architecture
    """
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    return model

def create_advanced_data_generators():
    """
    Create advanced data generators with extensive augmentation
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )
    
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )
    
    return train_datagen, validation_datagen

def create_callbacks(model_save_path):
    """
    Create advanced callbacks for better training
    """
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Learning rate scheduler
        LearningRateScheduler(
            lambda epoch: 0.001 * (0.9 ** epoch) if epoch < 20 else 0.0001
        )
    ]
    
    return callbacks

def train_enhanced_model(train_dir, validation_dir, model_type='transfer', 
                        batch_size=16, epochs=100, input_size=224):
    """
    Train the enhanced model with advanced techniques
    """
    logger.info(f"Training {model_type} model with input size {input_size}x{input_size}")
    
    # Create data generators
    train_datagen, validation_datagen = create_advanced_data_generators()
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    logger.info(f"Training samples: {train_generator.samples}")
    logger.info(f"Validation samples: {validation_generator.samples}")
    
    # Create model based on type
    if model_type == 'transfer':
        model, base_model = create_enhanced_model((input_size, input_size, 3))
        logger.info("Using transfer learning with ResNet50V2")
    else:
        model = create_custom_cnn_model((input_size, input_size, 3))
        logger.info("Using custom CNN architecture")
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    model_save_path = f'enhanced_brain_tumor_model_{model_type}.h5'
    callbacks = create_callbacks(model_save_path)
    
    # Phase 1: Train with frozen base model (if using transfer learning)
    if model_type == 'transfer':
        logger.info("Phase 1: Training with frozen base model...")
        history1 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks[:3],  # Exclude learning rate scheduler for phase 1
            verbose=1
        )
        
        # Phase 2: Fine-tune the entire model
        logger.info("Phase 2: Fine-tuning entire model...")
        base_model.trainable = True
        
        # Use a lower learning rate for fine-tuning
        optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        history2 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs - 20,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
    else:
        # Train custom model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    return model, history

def evaluate_model_performance(model, validation_generator):
    """
    Evaluate model performance with detailed metrics
    """
    # Evaluate on validation set
    evaluation = model.evaluate(validation_generator, verbose=1)
    
    # Get predictions
    predictions = model.predict(validation_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int)
    
    # Get true labels
    true_labels = validation_generator.classes
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    print(f"Loss: {evaluation[0]:.4f}")
    print(f"Accuracy: {evaluation[1]:.4f}")
    print(f"Precision: {evaluation[2]:.4f}")
    print(f"Recall: {evaluation[3]:.4f}")
    print(f"AUC: {evaluation[4]:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes, 
                              target_names=['No Tumor', 'Tumor']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predicted_classes)
    print(cm)
    
    # ROC AUC Score
    roc_auc = roc_auc_score(true_labels, predictions)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    return evaluation

if __name__ == "__main__":
    # Example usage
    print("Enhanced CNN Model for Brain Tumor Detection")
    print("This model is designed to achieve >90% accuracy")
