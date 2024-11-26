import os
import numpy as np
from PIL import Image
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define paths
MODEL_WEIGHTS_PATH = 'unet_road_segmentation.keras'  # Trained model weights
TEST_IMAGES_PATH = 'test_set_images/'                # Path to test images (with subfolders)
PREDICTED_MASKS_PATH = 'predicted_masks/'            # Directory to save predicted masks

# Parameters
IMG_HEIGHT = 400
IMG_WIDTH = 400
IMG_CHANNELS = 3

scaler = MinMaxScaler()

# Ensure the output directory exists
os.makedirs(PREDICTED_MASKS_PATH, exist_ok=True)

# Function to preprocess test images
def preprocess_images(image_paths, img_height, img_width):
    images = []
    for img_path in image_paths:
        with Image.open(img_path) as img:
            img = img.resize((img_width, img_height))  # Resize to model input size
            img = np.array(img) 
            # Normalize using MinMaxScaler
            img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        images.append(img)
    return np.array(images)

# Recursively collect all test images from subfolders
test_image_paths = sorted(glob(os.path.join(TEST_IMAGES_PATH, '**/*.png'), recursive=True))
if not test_image_paths:
    raise ValueError("No test images found in subfolders. Check your test set directory.")

print(f"Found {len(test_image_paths)} test images.")

# Preprocess test images
X_test = preprocess_images(test_image_paths, IMG_HEIGHT, IMG_WIDTH)

# Load the trained model
print("Loading the trained model...")
model = load_model(MODEL_WEIGHTS_PATH, compile=False)
print("Model loaded successfully.")

# Predict on test images
print("Predicting on test set...")
predictions = model.predict(X_test, batch_size=1, verbose=1)
predictions = (predictions > 0.5).astype(np.uint8)  # Binarize predictions

# Save predictions as PNG files
print("Saving predicted masks...")
for i, (img_path, pred) in enumerate(zip(test_image_paths, predictions)):
    # Construct output file name based on input image's folder and file name
    subfolder_name = os.path.basename(os.path.dirname(img_path))
    output_file_name = f'{subfolder_name}.png'  # Save as the subfolder's name
    pred_mask = (pred.squeeze() * 255).astype(np.uint8)  # Convert to uint8 format
    pred_image = Image.fromarray(pred_mask)  # Create a PIL Image
    pred_image.save(os.path.join(PREDICTED_MASKS_PATH, output_file_name))

print(f"Predicted masks saved to {PREDICTED_MASKS_PATH}.")
