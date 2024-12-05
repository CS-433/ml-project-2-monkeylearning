import sys
import os
import glob
from PIL import Image
from random import randint

CROP_SIZE = 400
SCALE_SIZE = 400

NUMBER_OF_EXTRA_SAMPLES = 3

IMAGES_PATH = "data/training/images/"
GROUNDTRUTH_PATH = "data/training/groundtruth/"
AUGMENTED_IMAGES_PATH = "data/training/augmented_images/"
AUGMENTED_GROUNDTRUTH_PATH = "data/training/augmented_groundtruth/"

def open_img(img_path):
    return Image.open(img_path)

def image_name_from_id(id, variation_id=""):
    return f'satImage_{id:03d}{"-" + str(variation_id) if variation_id != "" else ""}.png'

def image_path_from_id(id):
    return IMAGES_PATH + image_name_from_id(id)

def groundtruth_path_from_id(id):
    return GROUNDTRUTH_PATH + image_name_from_id(id)

def augmented_image_path_from_id(id, variation_id=""):
    return AUGMENTED_IMAGES_PATH + image_name_from_id(id, variation_id)

def augmented_groundtruth_path_from_id(id, variation_id=""):
    return AUGMENTED_GROUNDTRUTH_PATH + image_name_from_id(id, variation_id)

def img_centered_crop(img, new_width, new_height):
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    return img.crop((left, top, right, bottom))

def upscale(img, new_width, new_height):
    return img.resize((new_width, new_height), Image.NEAREST)

def generate_new_sample(img_path, orientation):
    img = open_img(img_path)
    return upscale(img_centered_crop(img.rotate(orientation), CROP_SIZE, CROP_SIZE), SCALE_SIZE, SCALE_SIZE)

def new_sample(id, variation_id):
    orientation = (variation_id)*90
    x = generate_new_sample(image_path_from_id(id), orientation)
    y = generate_new_sample(groundtruth_path_from_id(id), orientation)
    return x, y

def make_img_overlay(x, y):
    x = x.convert("RGB")
    y = y.convert("RGB")

    result = x.copy()

    pixels_result = result.load()
    pixels_y = y.load()

    for i in range(y.height):
        for j in range(y.width):
            if pixels_y[i, j] != (0, 0, 0):
                r, g, b = pixels_result[i, j]
                r = min(255, r + 60)
                pixels_result[i, j] = (r, g, b)

    return result

if __name__ == "__main__":
    if len(sys.argv) == 2:
        image_id = sys.argv[1].split("-")
        if len(image_id) == 1:
            id = int(image_id[0])
            img = open_img(image_path_from_id(id))
            groundtruth = open_img(groundtruth_path_from_id(id))
            make_img_overlay(img, groundtruth).show()
        else:
            id = int(image_id[0])
            variation_id = int(image_id[1])
            img = open_img(augmented_image_path_from_id(id, variation_id))
            groundtruth = open_img(augmented_groundtruth_path_from_id(id, variation_id))
            make_img_overlay(img, groundtruth).show()
        sys.exit()

    if os.path.exists(AUGMENTED_IMAGES_PATH):
        png_files = glob.glob(os.path.join(AUGMENTED_IMAGES_PATH, '*.png'))
        for file in png_files:
            os.remove(file)
    else:
        os.mkdir(AUGMENTED_IMAGES_PATH)

    if os.path.exists(AUGMENTED_GROUNDTRUTH_PATH):
        png_files = glob.glob(os.path.join(AUGMENTED_GROUNDTRUTH_PATH, '*.png'))
        for file in png_files:
            os.remove(file)     
    else:
        os.mkdir(AUGMENTED_GROUNDTRUTH_PATH)

    for id in range(1, 101):
        print(f"Current id: {id}")
        img = open_img(image_path_from_id(id))
        groundtruth = open_img(groundtruth_path_from_id(id))
        img = upscale(img, SCALE_SIZE, SCALE_SIZE)
        img.save(augmented_image_path_from_id(id))
        groundtruth = upscale(groundtruth, SCALE_SIZE, SCALE_SIZE)
        groundtruth.save(augmented_groundtruth_path_from_id(id))
        for variation_id in range(1, NUMBER_OF_EXTRA_SAMPLES + 1):
            new_img, new_groundtruth = new_sample(id, variation_id)
            new_img.save(augmented_image_path_from_id(id, variation_id))
            new_groundtruth.save(augmented_groundtruth_path_from_id(id, variation_id))