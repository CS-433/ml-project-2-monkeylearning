import sys
from PIL import Image

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

if len(sys.argv) >= 2:
    img = Image.open(f"data/test_set_images/test_{sys.argv[1]}/test_{sys.argv[1]}.png")
    mask = Image.open(f"data/predicted_masks/test_{sys.argv[1]}.png")
    make_img_overlay(img, mask).show()