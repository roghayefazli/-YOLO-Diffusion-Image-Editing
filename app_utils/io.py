# utils/io.py
from PIL import Image
import numpy as np
import os

def load_image(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)