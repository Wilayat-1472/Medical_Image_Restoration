import numpy as np
import torch
from typing import Dict, Any, Optional


class ConfidenceScorer:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def score(
        self,
        degraded: np.ndarray,
        restored: np.ndarray,
    ) -> Dict[str, Any]:
        degraded_gray = self._to_grayscale(degraded)
        restored_gray = self._to_grayscale(restored)

        diff = np.abs(restored_gray.astype(np.float32) - degraded_gray.astype(np.float32))
        local_std = self._local_std(restored_gray, window_size=8)
        edge_preservation = self._edge_preservation(degraded_gray, restored_gray)

        noise_reduction_conf = 1.0 - np.mean(diff) / (np.std(degraded_gray) + 1e-8)
        structure_conf = 1.0 - np.mean(local_std) / (np.max(local_std) + 1e-8)
        edge_conf = float(edge_preservation)

        overall = float(np.clip(
            0.4 * noise_reduction_conf + 0.3 * structure_conf + 0.3 * edge_conf,
            0.0, 1.0,
        ))

        return {
            "overall_confidence": overall,
            "method": "heuristic_image_consistency",
            "calibrated": False,
            "noise_reduction_confidence": float(np.clip(noise_reduction_conf, 0, 1)),
            "structure_preservation_confidence": float(np.clip(structure_conf, 0, 1)),
            "edge_preservation_confidence": edge_conf,
            "is_reliable": overall >= self.threshold,
        }

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[-1] == 3:
            return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return img.squeeze()

    def _local_std(self, img: np.ndarray, window_size: int = 8) -> np.ndarray:
        from skimage.util import view_as_windows
        h, w = img.shape
        h_adj = h - h % window_size
        w_adj = w - w % window_size
        img_cropped = img[:h_adj, :w_adj]
        windows = view_as_windows(img_cropped, (window_size, window_size), step=window_size)
        return np.std(windows, axis=(-2, -1))

    def _edge_preservation(self, degraded: np.ndarray, restored: np.ndarray) -> float:
        from skimage.filters import sobel
        edges_degraded = sobel(degraded)
        edges_restored = sobel(restored)
        if edges_degraded.max() > 0 and edges_restored.max() > 0:
            edges_degraded = edges_degraded / edges_degraded.max()
            edges_restored = edges_restored / edges_restored.max()
        diff = np.abs(edges_restored - edges_degraded)
        return float(1.0 - np.mean(diff))
