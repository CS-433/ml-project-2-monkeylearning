import os
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models as sm
import matplotlib.pyplot as plt
import logging

# ---------------------------
# Configuration
# ---------------------------
PATCHES_IMG_DIR = 'data/training/patches/images'
PATCHES_GT_DIR = 'data/training/patches/groundtruth'

PATCH_SIZE = 256
IMG_CHANNELS = 3
EPOCHS = 50
BATCH_SIZE = 4
SEED = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Load all patches
# ---------------------------
img_paths = sorted(glob(os.path.join(PATCHES_IMG_DIR, '*.png')))
mask_paths = sorted(glob(os.path.join(PATCHES_GT_DIR, '*.png')))

X = []
Y = []

for img_path, mask_path in zip(img_paths, mask_paths):
    img = np.array(Image.open(img_path)).astype('float32') / 255.0
    mask = np.array(Image.open(mask_path)).astype('float32') / 255.0
    # Ensure mask is [H,W,1]
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    X.append(img)
    Y.append(mask)

X = np.array(X)
Y = np.array(Y)

logging.info(f"Loaded {X.shape[0]} patches with shape {X.shape[1:]}")

# ---------------------------
# Train/Validation split
# ---------------------------
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=SEED)
logging.info(f'Training set size: {X_train.shape[0]}')
logging.info(f'Validation set size: {X_valid.shape[0]}')

# ---------------------------
# Define U-Net model
# ---------------------------
def unet_model(input_size=(PATCH_SIZE, PATCH_SIZE, IMG_CHANNELS)):
    inputs = Input(input_size)
    
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()

# ---------------------------
# Compile Model
# ---------------------------
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + focal_loss

model.compile(optimizer=Adam(), loss=total_loss, metrics=[sm.metrics.IOUScore()])

# ---------------------------
# Callbacks
# ---------------------------
checkpoint = ModelCheckpoint('weights.keras', verbose=1, save_best_only=True)
earlystop = EarlyStopping(patience=10, verbose=1)
callbacks_list = [checkpoint, earlystop]

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_valid, Y_valid),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

# ---------------------------
# Plot training history
# ---------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # IoU Score
    plt.subplot(1,2,1)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model IoU Score')
    plt.ylabel('IoU Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_training_history(history)
