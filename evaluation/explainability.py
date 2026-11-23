import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any

try:
    from matplotlib import pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: Optional[str] = None):
        layer = self._find_target_layer(target_layer)
        if layer is None:
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
                    if hasattr(module, "out_channels") and module.out_channels < 1024:
                        layer = module
                        break

        if layer is not None:
            self.handles = [layer.register_forward_hook(self._forward_hook),
                            layer.register_full_backward_hook(self._backward_hook)]

    def _find_target_layer(self, target_layer: Optional[str] = None):
        if target_layer is None:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    return module
            return None
        for name, module in self.model.named_modules():
            if name == target_layer:
                return module
        return None

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        output = self.model(x)
        if target_class is None:
            target = output.sum()
        else:
            target = output[0, target_class]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((x.shape[-2], x.shape[-1]))

        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (self.activations * gradients).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        for handle in getattr(self, "handles", []):
            handle.remove()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualize_explanations(
    model: torch.nn.Module,
    degraded: torch.Tensor,
    restored: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    if restored is None:
        restored = model(degraded)

    gradcam = GradCAM(model)
    cam = gradcam.generate(degraded)

    degraded_np = degraded.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    restored_np = restored.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

    degraded_np = (degraded_np * 0.5) + 0.5
    restored_np = (restored_np * 0.5) + 0.5
    degraded_np = np.clip(degraded_np, 0, 1)
    restored_np = np.clip(restored_np, 0, 1)

    result = {
        "degraded": degraded_np,
        "restored": restored_np,
        "attention_map": cam,
        "overlay": None,
    }

    if degraded_np.ndim == 3 and degraded_np.shape[-1] == 3:
        overlay = degraded_np.copy()
        for c in range(3):
            overlay[..., c] = overlay[..., c] * 0.6 + cam * 0.4
        result["overlay"] = np.clip(overlay, 0, 1)
    else:
        overlay = degraded_np.squeeze() * 0.6 + cam * 0.4
        result["overlay"] = np.clip(overlay, 0, 1)

    if save_path and _HAS_PLT:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Degraded", "Restored", "Attention Map", "Overlay"]
        images = [result["degraded"], result["restored"], result["attention_map"], result["overlay"]]
        for ax, title, img in zip(axes, titles, images):
            ax.imshow(img.squeeze(), cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return result
