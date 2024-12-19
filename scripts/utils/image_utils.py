import numpy as np
from PIL import Image

def load_image(image_path):
    return Image.open(image_path)

def save_image(image, save_path):
    image.save(save_path)

def resize_image(image, target_size=(256, 256)):
    return image.resize(target_size)

def mask_to_label(mask):
    # Convert mask to label (e.g., background = 0, land = 1, water = 2)
    return np.array(mask)  # For simplicity; use your actual mask-to-label logic here.
