import os
import numpy as np
import matplotlib.image as mpimg

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
IMG_PATCH_SIZE = 16
NUM_LABELS = 2

DATA_DIR = "data/training/"
TRAIN_DATA_DIR = DATA_DIR + "images/"
TRAIN_LABELS_DIR = DATA_DIR + "groundtruth/"

MODEL_DIR = "./saved_models/"

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

def build_balanced_data(y_train, x_train, total_samples = 200_000):
    positive_ids = np.where(np.argmax(y_train, axis=1) == 0)[0]
    negative_ids = np.where(np.argmax(y_train, axis=1) == 1)[0]

    required_samples_per_class = total_samples // 2

    if len(positive_ids) < required_samples_per_class:
        num_repeats = required_samples_per_class // len(positive_ids)
        remainder = required_samples_per_class % len(positive_ids)

        positive_ids_balanced = np.concatenate([positive_ids] * num_repeats + [positive_ids[:remainder]], axis=0)
    else:
        # Trim if there are more samples than required (not typical but handled)
        positive_ids_balanced = positive_ids[:required_samples_per_class]

    if len(negative_ids) < required_samples_per_class:
        num_repeats = required_samples_per_class // len(negative_ids)
        remainder = required_samples_per_class % len(negative_ids)

        negative_ids_balanced = np.concatenate([negative_ids] * num_repeats + [negative_ids[:remainder]], axis=0)
    else:
        # Trim if there are more samples than required (not typical but handled)
        negative_ids_balanced = negative_ids[:required_samples_per_class]

    ids_balanced = np.concatenate([positive_ids_balanced, negative_ids_balanced], axis=0)

    np.random.seed(1)
    np.random.shuffle(ids_balanced)

    x_balanced = x_train[ids_balanced]
    y_balanced = y_train[ids_balanced]

    return x_balanced, y_balanced

def load_data():

    # Extract the data
    training_size = 100
    train_data = extract_data(TRAIN_DATA_DIR, training_size)
    train_labels = extract_labels(TRAIN_LABELS_DIR, training_size)

    # Balance classes
    c0 = np.sum(np.argmax(train_labels, axis=1) == 0)
    c1 = np.sum(np.argmax(train_labels, axis=1) == 1)
    print(f"Class counts: c0 = {c0}, c1 = {c1}")

    train_data, train_labels = build_balanced_data(train_labels, train_data, 100_000)
    c0 = np.sum(np.argmax(train_labels, axis=1) == 0)
    c1 = np.sum(np.argmax(train_labels, axis=1) == 1)
    print(f"Class counts: c0 = {c0}, c1 = {c1}")

    print("Training data shape:", train_data.shape)
    print("Training labels shape:", train_labels.shape)

    return train_data, train_labels
