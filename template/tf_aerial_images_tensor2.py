"""
Baseline for machine learning project on road segmentation.
Updated for TensorFlow 2.x
"""

import os
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import tensorflow as tf

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
IMG_PATCH_SIZE = 16


# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    imgs = []
    for i in range(1, num_images + 1):
        imageid = f"satImage_{i:03d}"
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data, dtype=np.float32)


def value_to_class(v):
    foreground_threshold = 0.25
    return [0, 1] if np.sum(v) > foreground_threshold else [1, 0]


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = f"satImage_{i:03d}"
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    return labels.astype(np.float32)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same',
                               input_shape=(IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_LABELS, activation='softmax')
    ])
    return model


def main():
    data_dir = "training/"
    train_data_filename = data_dir + "images/"
    train_labels_filename = data_dir + "groundtruth/"

    # Extract the data
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    # Balance classes
    c0 = np.sum(np.argmax(train_labels, axis=1) == 0)
    c1 = np.sum(np.argmax(train_labels, axis=1) == 1)
    print(f"Class counts: c0 = {c0}, c1 = {c1}")

    min_c = min(c0, c1)
    idx0 = [i for i, label in enumerate(train_labels) if np.argmax(label) == 0]
    idx1 = [i for i, label in enumerate(train_labels) if np.argmax(label) == 1]
    balanced_indices = idx0[:min_c] + idx1[:min_c]
    train_data = train_data[balanced_indices]
    train_labels = train_labels[balanced_indices]

    print("Training data shape:", train_data.shape)
    print("Training labels shape:", train_labels.shape)

    # Build and compile the model
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # Save the model
    model.save("road_segmentation_model.h5")
    print("Model saved.")


if __name__ == "__main__":
    main()
