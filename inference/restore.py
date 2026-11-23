"""Validated restoration that retains the original image's spatial detail."""
from pathlib import Path
import cv2
import numpy as np
import torch
from data.preprocessing import MedicalTransform, denormalize
from data.io import load_image, unit_float
from .failsafe import FailSafe


class RestorationPipeline:
    def __init__(self, model, device=torch.device('cpu'), target_size=256, strength=0.5):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.target_size = target_size
        self.strength = self._validate_strength(strength)
        self.transform = MedicalTransform(target_size=(target_size, target_size), augment=False)
        self.validator = FailSafe()

    @staticmethod
    def _validate_strength(strength):
        strength = float(strength)
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Restoration strength must be between 0 and 1')
        return strength

    def _load_image(self, image):
        if isinstance(image, (str, Path)):
            return load_image(image)
        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError('Tensor input must be CHW in [0, 1]')
            image = image.detach().cpu().numpy().transpose(1, 2, 0)
        return unit_float(image)

    def _prepare(self, image):
        image = self._load_image(image)
        validation = self.validator.validate_input(image)
        if not validation['valid']:
            raise ValueError('; '.join(validation['issues']))
        return image

    @torch.inference_mode()
    def restore_batch(self, images, batch_size=4, strength=None):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        strength = self.strength if strength is None else self._validate_strength(strength)
        results = []
        for start in range(0, len(images), batch_size):
            originals = [self._prepare(img) for img in images[start:start + batch_size]]
            if not strength:
                results.extend({'restored': img.copy(), 'success': True, 'strength': 0.0,
                                'detail_preserving': True} for img in originals)
                continue
            batch = torch.stack([self.transform(img) for img in originals]).to(self.device)
            outputs = self.model(batch).float().cpu()
            for original, source_tensor, output in zip(originals, batch.cpu(), outputs):
                prediction = denormalize(output.numpy().transpose(1, 2, 0))
                # This is the precise low-resolution input used by the model.
                source = denormalize(source_tensor.float().numpy().transpose(1, 2, 0))
                correction = prediction - source
                if original.shape[-1] == 1:
                    correction = (0.299 * correction[..., 0] + 0.587 * correction[..., 1]
                                  + 0.114 * correction[..., 2])[..., None]
                h, w = original.shape[:2]
                if correction.shape[:2] != (h, w):
                    correction = cv2.resize(correction, (w, h), interpolation=cv2.INTER_LINEAR)
                    if correction.ndim == 2:
                        correction = correction[..., None]
                # Never replace a full-resolution image with an enlarged 128px
                # prediction. Apply its correction to the original pixel grid.
                # This retains detail above the model's resolution, including
                # high-frequency noise: it is not full-resolution denoising.
                restored = np.clip(original + strength * correction, 0, 1)
                results.append({'restored': unit_float(restored), 'success': True,
                                'strength': strength, 'detail_preserving': True})
        return results

    def restore(self, image, return_numpy=True, strength=None):
        result = self.restore_batch([image], batch_size=1, strength=strength)[0]
        if not return_numpy:
            result['restored_tensor'] = torch.from_numpy(result.pop('restored').transpose(2, 0, 1).copy())
        return result
