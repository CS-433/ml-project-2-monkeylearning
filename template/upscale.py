import os
from PIL import Image
from glob import glob

# Define paths
INPUT_FOLDER = 'predicted_masks_best/'  # Folder containing 400x400 images
OUTPUT_FOLDER = 'predicted_masks/'      # Folder to save 608x608 images

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all PNG files from the input folder
image_paths = sorted(glob(os.path.join(INPUT_FOLDER, '*.png')))

if not image_paths:
    raise ValueError("No images found in the input folder. Check the path and ensure it contains PNG files.")

print(f"Found {len(image_paths)} images to upscale.")

# Process each image
for img_path in image_paths:
    with Image.open(img_path) as img:
        # Check if the image is 400x400
        if img.size != (400, 400):
            print(f"Skipping {img_path}, as it is not 400x400.")
            continue

        # Resize the image to 608x608 using NEAREST neighbor interpolation
        upscaled_img = img.resize((608, 608), Image.NEAREST)

        # Save the upscaled image to the output folder
        output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
        upscaled_img.save(output_path)

print(f"Upscaled images saved to {OUTPUT_FOLDER}.")
