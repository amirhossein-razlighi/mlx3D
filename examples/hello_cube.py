from mlx3d.renderer import hello_image
from PIL import Image
import numpy as np
import os

img = hello_image(256, 256)
arr = np.array(img) if not isinstance(img, np.ndarray) else img
os.makedirs("outputs", exist_ok=True)
Image.fromarray((arr * 255).astype(np.uint8)).save("outputs/hello_example.png")
print("Saved outputs/hello_example.png")
