"""Reference metrics on [0, 1] images; normalized model tensors use [-1, 1]."""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


def compute_psnr(pred, target):
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def compute_ssim(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred, target = pred[..., 0], target[..., 0]
    side = min(pred.shape[:2])
    if side < 3:
        raise ValueError('SSIM requires images at least 3x3')
    win = min(7, side if side % 2 else side - 1)
    return float(structural_similarity(target, pred, channel_axis=-1 if pred.ndim == 3 else None,
                                       data_range=1.0, win_size=win))


def compute_mse(pred, target):
    return float(mean_squared_error(target, pred))


def compute_mae(pred, target):
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32))))


def _image(value, normalized):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
        if value.ndim != 3:
            raise ValueError('Metric tensors must be CHW, without a batch dimension')
        value = value.transpose(1, 2, 0)
    value = np.asarray(value, dtype=np.float32)
    if normalized:
        value = value * 0.5 + 0.5
    if not np.isfinite(value).all():
        raise ValueError('Metrics require finite images')
    return np.clip(value, 0, 1)


def compute_all_metrics(pred, target, normalized=None):
    if normalized is None:
        normalized = isinstance(pred, torch.Tensor)
    pred, target = _image(pred, normalized), _image(target, normalized)
    return {'psnr': compute_psnr(pred, target), 'ssim': compute_ssim(pred, target),
            'mse': compute_mse(pred, target), 'mae': compute_mae(pred, target)}
