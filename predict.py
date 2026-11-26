"""Local CPU inference script for Low-Dose CT restoration using a joblib bundle."""
import argparse
from pathlib import Path
import cv2
import joblib
import numpy as np
import torch

from src.model import ResidualAttentionUNet

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_bundle(bundle_path: str | Path) -> tuple[ResidualAttentionUNet, dict]:
    """Load ResidualAttentionUNet and metadata from a compressed joblib bundle on CPU."""
    bundle = joblib.load(bundle_path)
    cfg = dict(bundle.get("config", {}))
    if "img_size" in bundle and "image_size" not in cfg:
        cfg["image_size"] = bundle["img_size"]
    model = ResidualAttentionUNet(
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("out_channels", 1),
        base_c=cfg.get("base_c", cfg.get("base_filters", 32)),
    )
    state_dict = bundle.get("model_state_dict") or bundle.get("state_dict")
    if state_dict is None:
        raise KeyError("Bundle must contain 'model_state_dict' or 'state_dict'")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg


def load_ldct_slice(path: str | Path) -> np.ndarray:
    """Load a 2D LDCT slice (PNG/JPG/TIFF/BMP, .dcm, or .npy) normalized to float32 [0, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        import pydicom
        from pydicom.pixels import apply_modality_lut, apply_voi_lut

        ds = pydicom.dcmread(path)
        arr = apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds).astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        lo, hi = np.percentile(arr, [0.5, 99.5])
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(arr)
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        lo, hi = np.percentile(arr, [0.5, 99.5])
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(arr)
    if suffix not in RASTER_EXTENSIONS:
        raise ValueError(f"Unsupported input format: {suffix}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32) / 255.0


@torch.no_grad()
def restore_slice(model: ResidualAttentionUNet, image: np.ndarray,
                  target_size: int | None = None, strength: float = 1.0) -> np.ndarray:
    """Restore a [0, 1] 2D CT slice on CPU, preserving native resolution."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    h, w = image.shape[:2]
    work = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR) if target_size else image
    tensor = torch.from_numpy(work[None, None, ...]).float()
    pred = model(tensor).squeeze().cpu().numpy()
    if target_size and (h, w) != (target_size, target_size):
        correction = cv2.resize(pred - work, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(image + strength * correction, 0.0, 1.0).astype(np.float32)
    return np.clip(image + strength * (pred - image), 0.0, 1.0).astype(np.float32)


def save_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if not cv2.imwrite(str(path), u8):
        raise OSError(f"Failed to write {path}")


def main():
    parser = argparse.ArgumentParser(description="Local CPU inference for Low-Dose CT restoration")
    parser.add_argument("--input", required=True, help="Path to LDCT slice (.png, .jpg, .tif, .dcm, .npy)")
    parser.add_argument("--bundle", default="models/ldct_restorer_bundle.joblib", help="Path to joblib bundle")
    parser.add_argument("--output", default="restored.png", help="Output restored PNG path")
    parser.add_argument("--strength", type=float, default=1.0, help="Restoration blend strength in [0, 1]")
    parser.add_argument("--compare", default=None, help="Optional side-by-side comparison PNG path")
    args = parser.parse_args()

    model, cfg = load_bundle(args.bundle)
    ldct = load_ldct_slice(args.input)
    restored = restore_slice(model, ldct, target_size=cfg.get("image_size"), strength=args.strength)
    save_png(args.output, restored)
    print(f"Saved restored CT slice to {args.output}")

    if args.compare:
        side_by_side = np.concatenate([ldct, restored], axis=1)
        save_png(args.compare, side_by_side)
        print(f"Saved comparison image to {args.compare}")


if __name__ == "__main__":
    main()
