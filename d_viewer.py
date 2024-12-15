import os
import tkinter as tk
from tkinter import ttk
from glob import glob
from PIL import Image, ImageTk

from config import *

def make_img_overlay(x, y):
    x = x.convert("RGB")
    y = y.convert("RGB")

    result = x.copy()
    pixels_result = result.load()
    pixels_y = y.load()

    for i in range(y.height):
        for j in range(y.width):
            if pixels_y[j, i] != (0, 0, 0):
                r, g, b = pixels_result[j, i]
                r = min(255, r + 60)
                pixels_result[j, i] = (r, g, b)

    return result

def on_mouse_wheel(event, canvas):
    # Adjust scrolling for different platforms
    delta = event.delta if event.delta else -event.num  # Windows/Mac vs Linux
    canvas.yview_scroll(int(-delta / 120), "units")

def main(test_images_dir=TEST_IMAGES_DIR):
    # Find all test images
    image_paths = sorted(glob(os.path.join(test_images_dir, '**/*.png'), recursive=True))

    if not image_paths:
        print("No test images found.")
        return

    # Create a Tkinter window
    root = tk.Tk()
    root.title("All Predictions Viewer")
    root.geometry("1600x900")

    # Create a frame that will contain the canvas and scrollbar
    container = ttk.Frame(root)
    container.pack(fill='both', expand=True)

    # Create canvas and scrollbar
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Enable mouse wheel scrolling
    root.bind_all("<MouseWheel>", lambda e: on_mouse_wheel(e, canvas))  # Windows/macOS
    root.bind_all("<Button-4>", lambda e: on_mouse_wheel(e, canvas))  # Linux scroll up
    root.bind_all("<Button-5>", lambda e: on_mouse_wheel(e, canvas))  # Linux scroll down

    # Load all images and masks, create overlays, and display them
    images_per_row = 3
    thumbnail_size = (500, 500)  # Adjust as needed

    photo_images = []  # Keep references to PhotoImage objects to prevent garbage collection

    for idx, img_path in enumerate(image_paths):
        test_id = img_path.split('/')[-1].split('.')[0].split('_')[-1]
        mask_path = f"{PREDICTED_GROUNDTRUTH_DIR}/test_{test_id}.png"

        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
        except FileNotFoundError:
            continue

        overlay = make_img_overlay(img, mask)
        overlay = overlay.resize(thumbnail_size, resample=Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(overlay)
        photo_images.append(photo)

        # Determine grid position
        row = idx // images_per_row
        col = idx % images_per_row

        # Create a frame for each image with a label and place it in the grid
        frame = ttk.Frame(scrollable_frame, borderwidth=5, relief="groove")
        frame.grid(row=row, column=col, padx=5, pady=5)

        lbl = ttk.Label(frame, image=photo)
        lbl.pack()
        title_lbl = ttk.Label(frame, text=f"Test {test_id}")
        title_lbl.pack()

    root.mainloop()

if __name__ == '__main__':
    main()
