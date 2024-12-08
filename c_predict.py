import os
import numpy as np
from PIL import Image
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.morphology import binary_closing, disk, remove_small_objects, remove_small_holes
from skimage.util import img_as_ubyte

# Define paths
MODEL_WEIGHTS_PATH = 'weights.keras'  # Trained model weights
TEST_IMAGES_PATH = 'data/test_set_images/'                # Path to test images (with subfolders)
PREDICTED_MASKS_PATH = 'data/predicted_masks/'            # Directory to save predicted masks

# Parameters
IMG_HEIGHT = 400
IMG_WIDTH = 400
IMG_CHANNELS = 3

INPUT_SIZE = 608
PATCH_SIZE = 256
num_patches = (INPUT_SIZE + PATCH_SIZE - 1) // PATCH_SIZE

min_size = 40
hole_size = 20
structuring_element_size = 5
threshold = 0.5


scaler = MinMaxScaler()

# Ensure the output directory exists
os.makedirs(PREDICTED_MASKS_PATH, exist_ok=True)


def post_process_predictions(mask, min_size=min_size, hole_size=hole_size, structuring_element_size=structuring_element_size):
    """
    Post-process predictions to clean up the binary masks.

    Parameters:
        mask (ndarray): Input binary mask.
        min_size (int): Minimum size of connected components to retain.
        hole_size (int): Maximum size of holes to fill.
        structuring_element_size (int): Size of the disk-shaped structuring element for morphological operations.

    Returns:
        ndarray: Post-processed binary mask.
    """
    # Remove small objects
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    # Remove small holes
    mask = remove_small_holes(mask, area_threshold=hole_size)
    # Perform binary closing to smooth boundaries
    selem = disk(structuring_element_size)
    mask = binary_closing(mask, selem)
    return mask.astype(np.uint8)

def get_patch_positions(img_size, patch_size, num_patches):
    """
    Compute positions to place patches equally over the image.

    Parameters:
        img_size (int): Size of the image dimension (height or width).
        patch_size (int): Size of the patch dimension (height or width).
        num_patches (int): Number of patches along the dimension.

    Returns:
        List[int]: List of positions along the dimension.
    """
    positions = []
    for i in range(num_patches):
        pos = int(round(i * (img_size - patch_size) / (num_patches - 1)))
        positions.append(pos)
    return positions


def extract_patches_from_positions(img, patch_size, x_positions, y_positions):
    """
    Extract patches from the image using specified positions.

    Parameters:
        img (ndarray): The input image array.
        patch_size (int): The size of the patches.
        x_positions (List[int]): Positions along the x-axis.
        y_positions (List[int]): Positions along the y-axis.

    Returns:
        List[ndarray]: List of image patches.
    """
    patches = []
    for y in y_positions:
        for x in x_positions:
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches


def create_weighting_mask(patch_size):
    """
    Create a 2D weighting mask using a cosine window for smooth blending.

    Parameters:
        patch_size (int): The size of the patch.

    Returns:
        ndarray: A 2D weighting mask of shape (patch_size, patch_size).
    """
    # Create 1D cosine window
    cos_window = np.hanning(patch_size)
    # Create 2D window by outer product
    weight_mask = np.outer(cos_window, cos_window)
    return weight_mask


# Function to preprocess test images
def preprocess_image(image_path):
    with Image.open(image_path) as img:
        img = img.resize((INPUT_SIZE, INPUT_SIZE))  # Resize to model input size
        img = np.array(img)
        # Normalize using MinMaxScaler
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    return img


# Recursively collect all test images from subfolders
test_image_paths = sorted(glob(os.path.join(TEST_IMAGES_PATH, '**/*.png'), recursive=True))
if not test_image_paths:
    raise ValueError("No test images found in subfolders. Check your test set directory.")



print(f"Found {len(test_image_paths)} test images.")

# Load the trained model
print("Loading the trained model...")
model = load_model(MODEL_WEIGHTS_PATH, compile=False)
print("Model loaded successfully.")

# Create weighting mask
weight_mask = create_weighting_mask(PATCH_SIZE)


# Predict on test images
print("Predicting on test set...")
for img_path in test_image_paths:
    # Preprocess the image
    img = preprocess_image(img_path)

    # Initialize output and weight arrays
    output_image = np.zeros((INPUT_SIZE, INPUT_SIZE))
    weight_image = np.zeros((INPUT_SIZE, INPUT_SIZE))

    x_positions = get_patch_positions(INPUT_SIZE, PATCH_SIZE, num_patches)
    y_positions = get_patch_positions(INPUT_SIZE, PATCH_SIZE, num_patches)

    # Extract patches
    patches = []
    positions = []
    for y in y_positions:
        for x in x_positions:
            # Ensure we don't go out of bounds
            if y + PATCH_SIZE > INPUT_SIZE or x + PATCH_SIZE > INPUT_SIZE:
                continue
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            patches.append(patch)
            positions.append((y, x))

    patches = np.array(patches)

    # Predict on patches
    predictions = model.predict(patches, batch_size=1, verbose=0)
    predictions = predictions.squeeze()  # Remove extra dimensions if any

    # Reassemble the image with smooth blending
    for (y, x), pred_patch in zip(positions, predictions):
        # Apply weighting mask
        weighted_pred = pred_patch.squeeze() * weight_mask
        # Accumulate predictions
        output_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += weighted_pred
        # Accumulate weights
        weight_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += weight_mask

    # Avoid division by zero
    weight_image = np.where(weight_image == 0, 1, weight_image)
    # Normalize the output image
    output_image /= weight_image
    # Binarize predictions
    output_image = (output_image > threshold).astype(np.uint8)

    # Post-process the binary mask
    output_image = post_process_predictions(output_image)

    # Resize to original size if necessary (since we resized input images to INPUT_SIZE)
    pred_image = Image.fromarray((output_image * 255).astype(np.uint8))
    pred_image = pred_image.resize((608, 608), Image.NEAREST)

    # Construct output file name
    subfolder_name = os.path.basename(os.path.dirname(img_path))
    output_file_name = f'{subfolder_name}.png'
    # Save the predicted mask
    pred_image.save(os.path.join(PREDICTED_MASKS_PATH, output_file_name))

print(f"Predicted masks saved to {PREDICTED_MASKS_PATH}.")