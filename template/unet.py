# Import necessary libraries
import os
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define dataset paths
TRAIN_IMAGES_PATH = 'training/images/'
TRAIN_GROUNDTRUTH_PATH = 'training/groundtruth/'
TEST_IMAGES_PATH = 'test_set_images/'

# Define image dimensions and training parameters
IMG_HEIGHT = 400
IMG_WIDTH = 400
IMG_CHANNELS = 3
BATCH_SIZE = 4
EPOCHS = 50
SEED = 42

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Function to load and preprocess images
def load_images(image_paths, img_height, img_width):
    images = []
    for img_path in image_paths:
        with Image.open(img_path) as img:
            img = img.resize((img_width, img_height))
            img = np.array(img)
            # Normalize using MinMaxScaler
            img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        images.append(img)
    return np.array(images)

# Function to load and preprocess masks
def load_masks(mask_paths, img_height, img_width):
    masks = []
    for mask_path in mask_paths:
        with Image.open(mask_path) as mask:
            mask = mask.resize((img_width, img_height))
            mask = mask.convert('L')  # Convert to grayscale
            mask = np.array(mask)
            # Normalize using MinMaxScaler
            mask = scaler.fit_transform(mask.reshape(-1, 1)).reshape(mask.shape)
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        masks.append(mask)
    return np.array(masks)

# Load training images and masks
train_image_paths = sorted(glob(os.path.join(TRAIN_IMAGES_PATH, '*.png')))
train_mask_paths = sorted(glob(os.path.join(TRAIN_GROUNDTRUTH_PATH, '*.png')))

logging.info(f'Number of training images: {len(train_image_paths)}')
logging.info(f'Number of training masks: {len(train_mask_paths)}')

X = load_images(train_image_paths, IMG_HEIGHT, IMG_WIDTH)
Y = load_masks(train_mask_paths, IMG_HEIGHT, IMG_WIDTH)

# Binarize masks
Y = (Y > 0.5).astype(np.float32)

# Split data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.1, random_state=SEED
)

logging.info(f'Training set size: {X_train.shape[0]}')
logging.info(f'Validation set size: {X_valid.shape[0]}')

# Define U-Net model with Conv2DTranspose and softmax activation
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)
    
    # Contracting path
    c1 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    c4 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2,2))(c4)
    
    # Bottleneck (maximum 512 filters)
    c5 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expansive path with Conv2DTranspose
    u6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    # Use softmax activation for multi-class segmentation
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Instantiate U-Net model
model = unet_model()

# Define Dice Loss and Focal Loss
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + focal_loss

# Compile the model
model.compile(optimizer=Adam(), loss=total_loss, metrics=[sm.metrics.IOUScore()])

# Define callbacks
checkpoint = ModelCheckpoint(
    'unet_road_segmentation.keras', verbose=1, save_best_only=True
)
earlystop = EarlyStopping(patience=10, verbose=1)
callbacks_list = [checkpoint, earlystop]

# Train the model
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_valid, Y_valid),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

# Plot training history
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

# Evaluate on validation set
model.load_weights('unet_road_segmentation.keras')
val_loss, val_iou = model.evaluate(X_valid, Y_valid)
logging.info(f'Validation Loss: {val_loss}')
logging.info(f'Validation IoU Score: {val_iou}')

# Load test images
test_image_paths = sorted(glob(os.path.join(TEST_IMAGES_PATH, 'test_*/*.png')))
X_test = load_images(test_image_paths, IMG_HEIGHT, IMG_WIDTH)

# Predict on test set
predictions = model.predict(X_test, verbose=1)
predictions = (predictions > 0.5).astype(np.uint8)

# Directory to save predicted masks
PREDICTED_MASKS_PATH = 'predicted_masks/'
os.makedirs(PREDICTED_MASKS_PATH, exist_ok=True)

# Save predicted masks using PIL
for i, pred in enumerate(predictions):
    pred_mask = (pred.squeeze() * 255).astype(np.uint8)  # Convert to uint8
    pred_image = Image.fromarray(pred_mask)  # Create a PIL Image
    pred_image.save(os.path.join(PREDICTED_MASKS_PATH, f'pred_mask_{i+1:03d}.png'))
