import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from tqdm import tqdm

from data.degradation import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_motion_blur,
    add_gaussian_blur,
    add_speckle_noise,
    adjust_contrast,
)
from data.preprocessing import MedicalTransform
from evaluation.metrics import compute_all_metrics


class EdgeCaseTester:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        target_size: int = 256,
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.transform = MedicalTransform(target_size=(target_size, target_size), augment=False)

    @torch.no_grad()
    def test_noise_levels(
        self, clean_images: List[np.ndarray], noise_levels: List[int] = [5, 10, 20, 30, 50, 70]
    ) -> Dict:
        results = {}
        for std in noise_levels:
            metrics_list = []
            for img in clean_images:
                degraded = add_gaussian_noise(img, std=std)
                degraded_t = self.transform(degraded).unsqueeze(0).to(self.device)
                clean_t = self.transform(img).unsqueeze(0).to(self.device)
                restored = self.model(degraded_t)
                m = compute_all_metrics(restored.squeeze(0), clean_t.squeeze(0))
                metrics_list.append(m)
            results[f"gaussian_noise_std_{std}"] = {
                k: float(np.mean([v for v in [m[k] for m in metrics_list]]))
                for k in metrics_list[0]
            }
        return results

    @torch.no_grad()
    def test_blur_kernels(
        self, clean_images: List[np.ndarray], kernel_sizes: List[int] = [3, 5, 7, 11, 15, 21, 31]
    ) -> Dict:
        results = {}
        for ksize in kernel_sizes:
            metrics_list = []
            for img in clean_images:
                degraded = add_motion_blur(img, kernel_size=ksize, angle=np.random.uniform(0, 360))
                degraded_t = self.transform(degraded).unsqueeze(0).to(self.device)
                clean_t = self.transform(img).unsqueeze(0).to(self.device)
                restored = self.model(degraded_t)
                m = compute_all_metrics(restored.squeeze(0), clean_t.squeeze(0))
                metrics_list.append(m)
            results[f"motion_blur_kernel_{ksize}"] = {
                k: float(np.mean([v for v in [m[k] for m in metrics_list]]))
                for k in metrics_list[0]
            }
        return results

    @torch.no_grad()
    def test_mixed_degradation(
        self, clean_images: List[np.ndarray], num_samples: int = 50
    ) -> Dict:
        results = []
        for i in range(min(num_samples, len(clean_images))):
            img = clean_images[i]
            degraded = add_gaussian_noise(img, std=np.random.randint(10, 40))
            degraded = add_motion_blur(degraded, kernel_size=np.random.choice([5, 9, 15]))
            degraded = adjust_contrast(degraded, factor=np.random.uniform(0.4, 0.8))

            degraded_t = self.transform(degraded).unsqueeze(0).to(self.device)
            clean_t = self.transform(img).unsqueeze(0).to(self.device)
            restored = self.model(degraded_t)
            m = compute_all_metrics(restored.squeeze(0), clean_t.squeeze(0))
            results.append(m)

        avg = {}
        for k in results[0]:
            avg[k] = float(np.mean([r[k] for r in results]))
        return {"mixed_degradation": avg}

    @torch.no_grad()
    def test_salt_pepper(
        self, clean_images: List[np.ndarray], amounts: List[float] = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5]
    ) -> Dict:
        results = {}
        for amt in amounts:
            metrics_list = []
            for img in clean_images:
                degraded = add_salt_pepper_noise(img, amount=amt)
                degraded_t = self.transform(degraded).unsqueeze(0).to(self.device)
                clean_t = self.transform(img).unsqueeze(0).to(self.device)
                restored = self.model(degraded_t)
                m = compute_all_metrics(restored.squeeze(0), clean_t.squeeze(0))
                metrics_list.append(m)
            results[f"salt_pepper_{amt:.2f}"] = {
                k: float(np.mean([v for v in [m[k] for m in metrics_list]]))
                for k in metrics_list[0]
            }
        return results

    def run_all(
        self, clean_images: List[np.ndarray]
    ) -> Dict[str, Dict]:
        if not clean_images:
            raise ValueError("Edge-case tests require reference images")
        results = {}
        print("Testing noise levels...")
        results.update(self.test_noise_levels(clean_images))
        print("Testing blur kernels...")
        results.update(self.test_blur_kernels(clean_images))
        print("Testing salt & pepper noise...")
        results.update(self.test_salt_pepper(clean_images))
        print("Testing mixed degradation...")
        results.update(self.test_mixed_degradation(clean_images))
        return results
