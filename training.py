import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models as sm
import matplotlib.pyplot as plt
import logging

from model import *
from config import *

# ---------------------------
# Configuration
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(show_train_graph=False):
    # ---------------------------
    # Load all patches
    # ---------------------------
    img_paths = sorted(glob(os.path.join(PATCHES_IMG_DIR, '*.png')))
    mask_paths = sorted(glob(os.path.join(PATCHES_GROUNDTRUTH_DIR, '*.png')))

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
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=RANDOM_SEED)
    logging.info(f'Training set size: {X_train.shape[0]}')
    logging.info(f'Validation set size: {X_valid.shape[0]}')

    # ---------------------------
    # Compile Model
    # ---------------------------
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + focal_loss

    model = unet_model()
    model.compile(optimizer=Adam(), loss=total_loss, metrics=[sm.metrics.IOUScore()])

    # ---------------------------
    # Callbacks
    # ---------------------------
    checkpoint = ModelCheckpoint(MODEL_WEIGHTS_PATH, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(patience=10, verbose=1)
    callbacks_list = [checkpoint, earlystop]

    # ---------------------------
    # Train Model
    # ---------------------------
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_valid, Y_valid),
        batch_size=TRAIN_BATCH_SIZE,
        epochs=TRAIN_EPOCHS,
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

    if show_train_graph:
        plot_training_history(history)

if __name__ == "__main__":
    main(True)