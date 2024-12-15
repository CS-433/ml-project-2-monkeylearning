#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
patch_size = 16  # Size of each patch (16x16)


def patch_to_label(patch):
    """Assign a label to a patch based on the foreground threshold."""
    df = np.mean(patch)
    return 1 if df > foreground_threshold else 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings for the submission file."""
    print(f"Processing file: {image_filename}")
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    
    # Print image dimensions for debugging
    print(f"Image shape: {im.shape} (H x W)")

    # Ensure the image dimensions are divisible by the patch size
    height, width = im.shape[0], im.shape[1]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Image dimensions {height}x{width} are not divisible by patch size {patch_size}.")

    row_count = 0
    for j in range(0, width, patch_size):
        for i in range(0, height, patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            row_count += 1
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))
    
    # Print number of rows generated for this file
    print(f"Generated {row_count} rows for file {image_filename}")


def masks_to_submission(submission_filename, *image_filenames):
    """Creates the submission file from all mask files."""
    print(f"Creating submission file: {submission_filename}")
    total_rows = 0
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in sorted(image_filenames):
            for s in mask_to_submission_strings(fn):
                f.write(f"{s}\n")
                total_rows += 1
    # Print total number of rows in submission
    print(f"Total rows written to {submission_filename}: {total_rows}")


def main():
    submission_filename = 'ml_submission.csv'
    predicted_masks_folder = 'data/predicted_masks'
    
    # Collect all predicted mask file paths
    image_filenames = sorted(
        [os.path.join(predicted_masks_folder, f) for f in os.listdir(predicted_masks_folder) if f.endswith('.png')]
    )
    
    # Debug: Print file list
    print(f"Found {len(image_filenames)} files in {predicted_masks_folder}:")
    for fn in image_filenames:
        print(f"  {fn}")
    
    # Check if the correct number of files are present
    if len(image_filenames) != 50:
        raise ValueError(f"Expected 50 files, but found {len(image_filenames)}.")
    
    # Create the submission file
    masks_to_submission(submission_filename, *image_filenames)

if __name__ == '__main__':
    main()