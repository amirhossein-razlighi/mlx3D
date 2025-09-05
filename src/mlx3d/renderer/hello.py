import numpy as np
try:
    import mlx.core as mx
except Exception:
    mx = None  # allow import on non-Apple platforms for now

def hello_image(h: int = 128, w: int = 128):
    """
    Return a simple RGB gradient image using MLX if available, else numpy fallback.
    """
    if mx is not None:
        xs = mx.repeat(mx.linspace(0.0, 1.0, w)[None, :], h, axis=0)
        ys = mx.repeat(mx.linspace(0.0, 1.0, h)[:, None], w, axis=1)
        img = mx.stack([xs, ys, mx.ones_like(xs)*0.25], axis=-1)
        return img  # MLX array
    else:
        xs = np.linspace(0.0, 1.0, w)[None, :].repeat(h, axis=0)
        ys = np.linspace(0.0, 1.0, h)[:, None].repeat(w, axis=1)
        img = np.stack([xs, ys, np.ones_like(xs)*0.25], axis=-1)
        return img  # numpy array

def main():
    """
    Console script entry: `mlx3d-hello`
    Saves outputs/hello.png
    """
    import os
    from PIL import Image

    img = hello_image(256, 256)
    arr = np.array(img) if not isinstance(img, np.ndarray) else img
    os.makedirs("outputs", exist_ok=True)
    Image.fromarray((arr * 255).astype(np.uint8)).save("outputs/hello.png")
    print("Saved outputs/hello.png")
