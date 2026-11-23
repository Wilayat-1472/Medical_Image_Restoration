import numpy as np
from typing import Dict, Any, Optional, Tuple


class FailSafe:
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def validate_input(self, image: np.ndarray) -> Dict[str, Any]:
        issues = []
        warnings = []

        if image.size == 0:
            return {"valid": False, "issues": ["Empty image"], "warnings": [], "can_proceed": False}
        if image.ndim not in (2, 3):
            return {"valid": False, "issues": ["Expected a 2D image"], "warnings": [], "can_proceed": False}
        if image.ndim == 2:
            pass
        elif image.ndim == 3:
            if image.shape[-1] not in [1, 3, 4]:
                issues.append(f"Unsupported channels: {image.shape[-1]}")

        if image.size == 0:
            issues.append("Empty image")

        if image.dtype not in [np.float32, np.float64, np.uint8, np.uint16]:
            warnings.append(f"Unusual dtype: {image.dtype}")

        if np.isnan(image).any() or np.isinf(image).any():
            issues.append("Image contains NaN or Inf values")

        max_val = image.max()
        min_val = image.min()
        if max_val == min_val:
            issues.append("Image has zero dynamic range (all pixels identical)")

        if image.ndim >= 2:
            h, w = image.shape[:2]
            if h < 32 or w < 32:
                issues.append(f"Image too small: {h}x{w} (minimum 32x32)")
            if h > 4096 or w > 4096:
                warnings.append(f"Image very large: {h}x{w}")

        intensity_range = max_val - min_val
        if intensity_range < 0.01 and max_val <= 1.0:
            warnings.append("Very low contrast image")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "can_proceed": len(issues) == 0,
        }

    def check_restoration_quality(
        self,
        restored: np.ndarray,
        confidence_score: float,
        quality_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        result = {
            "passes": True,
            "warnings": [],
            "recommendations": [],
        }

        if confidence_score < self.confidence_threshold:
            result["passes"] = False
            result["warnings"].append(
                f"Low restoration confidence: {confidence_score:.3f} (threshold: {self.confidence_threshold})"
            )
            result["recommendations"].append("Manual expert review required")
            result["recommendations"].append("Do not use for clinical decision-making")

        restored_norm = restored.astype(np.float32)
        if restored_norm.max() > 1.0 or restored_norm.min() < -0.1:
            result["warnings"].append("Restored image outside expected intensity range")
            result["recommendations"].append("Verify image normalization")

        local_std_map = self._compute_local_std(restored_norm, window=4)
        artifact_score = float(np.mean(local_std_map > 0.3))

        if artifact_score > 0.1:
            result["passes"] = False
            result["warnings"].append(f"Possible restoration artifacts detected (score: {artifact_score:.3f})")
            result["recommendations"].append("Review for hallucinated structures")

        return result

    def _compute_local_std(self, img: np.ndarray, window: int = 4) -> np.ndarray:
        if img.ndim == 3:
            img = np.mean(img, axis=-1)
        from skimage.util import view_as_windows
        h, w = img.shape
        h_adj = h - h % window
        w_adj = w - w % window
        img_cropped = img[:h_adj, :w_adj]
        windows = view_as_windows(img_cropped, (window, window), step=window // 2)
        return np.std(windows, axis=(-2, -1))
