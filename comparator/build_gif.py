from PIL import Image
from pathlib import Path


def create_gif(image_paths, output_gif_path, duration=500):
    """Build an animated GIF from a sequence of image files.

    Parameters
    ----------
    image_paths : list of str or Path
        Ordered list of image file paths (e.g. PNGs) to combine.
    output_gif_path : str or Path
        Destination path for the output GIF.
    duration : int
        Duration per frame in milliseconds (default 500).
    """
    if not image_paths:
        raise ValueError("No image paths provided for GIF creation.")

    images = [Image.open(p) for p in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # infinite loop
    )
    return Path(output_gif_path)
