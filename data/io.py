"""Read 2D raster images into RGB/grayscale float32 [0, 1]."""
from pathlib import Path
import cv2
import numpy as np

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def unit_float(image):
    image = np.asarray(image)
    if image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Image must be nonempty and finite")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        image = image.astype(np.float32)
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Floating-point images must be normalized to [0, 1]")
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC 2D image, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def load_image(path):
    if Path(path).suffix.lower() not in RASTER_EXTENSIONS:
        raise ValueError("Convert medical volumes with scripts/prepare_data.py before restoration")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
    return unit_float(image)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (unit_float(image) * 255).round().astype(np.uint8)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = image[..., 0]
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")
