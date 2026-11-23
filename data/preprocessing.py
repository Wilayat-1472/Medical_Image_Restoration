import cv2
import numpy as np
from typing import Tuple, Optional, List
from .constants import DEFAULT_MEAN, DEFAULT_STD
from .io import unit_float
import torch


def normalize(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    if mean is None:
        mean = DEFAULT_MEAN
    if std is None:
        std = DEFAULT_STD
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    for c in range(image.shape[-1]):
        image[..., c] = (image[..., c] - mean[c]) / (std[c] + 1e-8)
    return image


def denormalize(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    if mean is None:
        mean = DEFAULT_MEAN
    if std is None:
        std = DEFAULT_STD
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    for c in range(image.shape[-1]):
        image[..., c] = image[..., c] * std[c] + mean[c]
    return np.clip(image, 0.0, 1.0)


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    if image.ndim == 2:
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
    h, w = image.shape[:2]
    if (h, w) == target_size:
        return image
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.equalizeHist((image * 255).astype(np.uint8)).astype(np.float32) / 255.0
    if image.shape[-1] == 1:
        return cv2.equalizeHist((image[..., 0] * 255).astype(np.uint8)).astype(np.float32)[..., None] / 255.0
    img_yuv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    img_yuv[..., 0] = cv2.equalizeHist(img_yuv[..., 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0


class MedicalTransform:
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        histogram_eq: bool = False,
        augment: bool = False,
    ):
        self.target_size = target_size
        self.do_normalize = normalize
        self.do_histogram_eq = histogram_eq
        self.augment = augment

    def __call__(self, image: np.ndarray, rng=None) -> torch.Tensor:
        image = unit_float(image)

        if self.do_histogram_eq:
            image = histogram_equalization(image)

        image = resize_image(image, self.target_size)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if image.shape[-1] == 1 and self.do_normalize:
            image = np.repeat(image, 3, axis=-1)

        if self.augment:
            image = self._augment(image, rng)

        image = np.clip(image, 0.0, 1.0)

        image = image.transpose(2, 0, 1)

        if self.do_normalize:
            mean = np.array(DEFAULT_MEAN, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(DEFAULT_STD, dtype=np.float32).reshape(-1, 1, 1)
            image = (image - mean) / (std + 1e-8)

        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))

    def _augment(self, image: np.ndarray, rng=None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        if rng.random() > 0.5:
            image = np.fliplr(image)
        if rng.random() > 0.5:
            image = np.flipud(image)
        if rng.random() > 0.5:
            k = rng.choice([1, 3])
            image = np.rot90(image, k=k)
        return image
