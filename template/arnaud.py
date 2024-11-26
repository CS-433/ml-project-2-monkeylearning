import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
IMG_PATCH_SIZE = 64  # Patch size
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 2
DATA_DIR = "training/"
SEED = 42

# Functions to load and preprocess data
def load_images_and_labels(img_dir, label_dir):
    """
    Load images and their corresponding labels as numpy arrays.
    """
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')])

    images = [tf.image.decode_png(tf.io.read_file(f), channels=3).numpy() for f in img_files]
    labels = [tf.image.decode_png(tf.io.read_file(f), channels=1).numpy() for f in label_files]
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels, patch_size):
    """
    Extract patches from images and labels.
    """
    def extract_patches(img, patch_size):
        patches = tf.image.extract_patches(
            images=tf.expand_dims(img, axis=0),
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, (-1, patch_size, patch_size, img.shape[-1]))
        return patches.numpy()
    
    image_patches = np.concatenate([extract_patches(img, patch_size) for img in images], axis=0)
    label_patches = np.concatenate([extract_patches(lbl, patch_size) for lbl in labels], axis=0)
    
    # Convert labels to binary class (road=1, background=0)
    label_patches = (np.mean(label_patches, axis=(1, 2, 3)) > 0.25).astype(int)
    
    return image_patches, label_patches

# Data Augmentation
def get_data_generators(X_train, y_train):
    """
    Create data augmentation generators for training.
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2
    )
    train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    return train_gen

# CNN Model
def create_cnn_model(input_shape):
    """
    Create a CNN model for binary classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    images, labels = load_images_and_labels(DATA_DIR + "images/", DATA_DIR + "groundtruth/")
    X, y = preprocess_data(images, labels, IMG_PATCH_SIZE)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    # Data Augmentation
    train_gen = get_data_generators(X_train, y_train)
    
    # Create and train model
    input_shape = (IMG_PATCH_SIZE, IMG_PATCH_SIZE, 3)
    model = create_cnn_model(input_shape)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr],
        steps_per_epoch=len(X_train) // BATCH_SIZE
    )
    
    # Save model
    model.save("road_segmentation_cnn.h5")
    print("Model saved as road_segmentation_cnn.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy")
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Loss")
    
    plt.show()
