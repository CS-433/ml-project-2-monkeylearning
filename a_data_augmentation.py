import os
import numpy as np
from PIL import Image
from glob import glob
from random import randint
from sklearn.preprocessing import MinMaxScaler
import logging
from skimage.color import rgb2hsv, hsv2rgb

# ---------------------------
# Configuration
# ---------------------------
TRAIN_IMAGES_PATH = 'data/training/images/'
TRAIN_GROUNDTRUTH_PATH = 'data/training/groundtruth/'
PATCHES_IMG_DIR = 'data/training/patches/images'
PATCHES_GT_DIR = 'data/training/patches/groundtruth'

IMG_HEIGHT = 400
IMG_WIDTH = 400
PATCH_SIZE = 256
IMG_CHANNELS = 3
NUM_IMAGES = 100
NUM_RANDOM_ANGLES = 5
overlap = PATCH_SIZE - (IMG_HEIGHT - PATCH_SIZE)  # Should be 112

scaler = MinMaxScaler()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Helper functions
# ---------------------------

def clear_directory(directory):
    """Remove all files in the given directory."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        logging.info(f"Cleared directory: {directory}")
    else:
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def image_name_from_id(id):
    return f"satImage_{id:03d}.png"

def load_original_image(id, size=IMG_HEIGHT):
    img_path = os.path.join(TRAIN_IMAGES_PATH, image_name_from_id(id))
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.NEAREST)
    return img

def load_original_mask(id, size=IMG_HEIGHT):
    gt_path = os.path.join(TRAIN_GROUNDTRUTH_PATH, image_name_from_id(id))
    gt = Image.open(gt_path).convert("L")
    gt = gt.resize((size, size), Image.NEAREST)
    return gt

def rotate_image(img, angle):
    return img.rotate(angle, resample=Image.NEAREST, expand=False)

def adjust_hsv(img_array):
    """
    Adjust hue, saturation, and brightness of the given RGB image array (values in [0,1]).
    We'll apply small random variations.
    """
    hue_shift_range = 0.02
    sat_scale_range = (0.9, 1.1)
    val_scale_range = (0.9, 1.1)
    
    hsv = rgb2hsv(img_array)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    dh = np.random.uniform(-hue_shift_range, hue_shift_range)
    sf = np.random.uniform(sat_scale_range[0], sat_scale_range[1])
    vf = np.random.uniform(val_scale_range[0], val_scale_range[1])
    
    h = (h + dh) % 1.0
    s = np.clip(s * sf, 0, 1)
    v = np.clip(v * vf, 0, 1)
    
    hsv_adjusted = np.stack([h, s, v], axis=-1)
    rgb_adjusted = hsv2rgb(hsv_adjusted)
    return rgb_adjusted

def get_positions(img_size, patch_size, step_size):
    positions = list(range(0, img_size - patch_size + 1, step_size))
    if positions[-1] != img_size - patch_size:
        positions.append(img_size - patch_size)
    return positions

def extract_patches(img, patch_size, overlap):
    img_h, img_w = img.shape[:2]
    step_size = patch_size - overlap
    x_positions = get_positions(img_w, patch_size, step_size)
    y_positions = get_positions(img_h, patch_size, step_size)
    patches = []
    for y in y_positions:
        for x in x_positions:
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def binarize_masks(Y):
    return (Y > 0.5).astype(np.float32)


# Clear the folders before use
clear_directory(PATCHES_IMG_DIR)
clear_directory(PATCHES_GT_DIR)

# Ensure directories exist
if not os.path.exists(PATCHES_IMG_DIR):
    os.makedirs(PATCHES_IMG_DIR)
if not os.path.exists(PATCHES_GT_DIR):
    os.makedirs(PATCHES_GT_DIR)

# ---------------------------
# Load original data
# ---------------------------
X_original = []
Y_original = []
for i in range(1, NUM_IMAGES + 1):
    original_img = load_original_image(i, size=IMG_HEIGHT)
    original_img = np.array(original_img).astype('float32') / 255.0
    # Adjust HSV
    original_img = adjust_hsv(original_img)
    
    original_mask = load_original_mask(i, size=IMG_HEIGHT)
    original_mask = np.array(original_mask).astype('float32') / 255.0
    
    X_original.append(original_img)
    Y_original.append(original_mask)

X_original = np.array(X_original)
Y_original = np.array(Y_original)

# ---------------------------
# Create patches from main rotations (0°,90°,180°,270°)
# ---------------------------
angles_main = [0, 90]
X_main = []
Y_main = []

for i in range(NUM_IMAGES):
    img = (X_original[i]*255).astype('uint8')   # convert back to uint8 for PIL rotation
    mask = (Y_original[i]*255).astype('uint8')
    
    img_pil = Image.fromarray(img)
    mask_pil = Image.fromarray(mask)
    
    for angle in angles_main:
        rotated_img = rotate_image(img_pil, angle)
        rotated_mask = rotate_image(mask_pil, angle)
        
        rotated_img = np.array(rotated_img).astype('float32') / 255.0
        rotated_mask = np.array(rotated_mask).astype('float32') / 255.0
        
        # Normalize (already 0-1, but we use scaler to be consistent)
        rotated_img = scaler.fit_transform(rotated_img.reshape(-1, IMG_CHANNELS)).reshape(rotated_img.shape)
        rotated_mask = scaler.fit_transform(rotated_mask.reshape(-1, 1)).reshape(rotated_mask.shape)

        img_patches = extract_patches(rotated_img, PATCH_SIZE, overlap)
        mask_patches = extract_patches(rotated_mask, PATCH_SIZE, overlap)
        
        for ip, mp in zip(img_patches, mask_patches):
            X_main.append(ip)
            Y_main.append(mp)

X_main = np.array(X_main)
Y_main = np.array(Y_main)
Y_main = binarize_masks(Y_main)

# ---------------------------
# Create patches from random angles
# ---------------------------
X_random = []
Y_random = []


for i in range(NUM_IMAGES):
    img = (X_original[i]*255).astype('uint8')
    mask = (Y_original[i]*255).astype('uint8')
    
    img_pil = Image.fromarray(img)
    mask_pil = Image.fromarray(mask)
    
    for angle in range(NUM_RANDOM_ANGLES):
        random_angle = randint(2, 357)
        rotated_img = rotate_image(img_pil, angle)
        rotated_mask = rotate_image(mask_pil, angle)
        
        # Extract center 256x256 patch
        x0 = (IMG_WIDTH - PATCH_SIZE)//2
        y0 = (IMG_HEIGHT - PATCH_SIZE)//2
        x1 = x0 + PATCH_SIZE
        y1 = y0 + PATCH_SIZE

        img_patch = rotated_img.crop((x0, y0, x1, y1))
        mask_patch = rotated_mask.crop((x0, y0, x1, y1))

        img_patch = np.array(img_patch).astype('float32') / 255.0
        mask_patch = np.array(mask_patch).astype('float32') / 255.0

        img_patch = scaler.fit_transform(img_patch.reshape(-1, IMG_CHANNELS)).reshape(img_patch.shape)
        mask_patch = scaler.fit_transform(mask_patch.reshape(-1, 1)).reshape(mask_patch.shape)

        X_random.append(img_patch)
        Y_random.append(mask_patch)

X_random = np.array(X_random)
Y_random = np.array(Y_random)
Y_random = binarize_masks(Y_random)

# ---------------------------
# Combine all patches
# ---------------------------
X_combined = np.concatenate((X_main, X_random), axis=0)
Y_combined = np.concatenate((Y_main, Y_random), axis=0)

logging.info(f"Total patches: {X_combined.shape[0]}")

# ---------------------------
# Save patches to disk
# ---------------------------
count = 0
for img_patch, mask_patch in zip(X_combined, Y_combined):
    # Convert back to uint8 for saving
    img_patch_uint8 = (img_patch*255).astype('uint8')
    mask_patch_uint8 = (mask_patch*255).astype('uint8').squeeze()

    img_pil = Image.fromarray(img_patch_uint8)
    mask_pil = Image.fromarray(mask_patch_uint8)

    img_path = os.path.join(PATCHES_IMG_DIR, f"img_{count:04d}.png")
    mask_path = os.path.join(PATCHES_GT_DIR, f"mask_{count:04d}.png")

    img_pil.save(img_path)
    mask_pil.save(mask_path)

    count += 1

logging.info("All patches saved successfully.")
