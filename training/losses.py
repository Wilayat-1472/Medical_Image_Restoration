import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))


class PerceptualLoss(nn.Module):
    def __init__(self, layers: list = [3, 8, 17, 26]):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.blocks.append(vgg[prev:layer])
            prev = layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # The restoration network uses [-1, 1]; ImageNet VGG expects standardized RGB.
        mean = pred.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = pred.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        loss = 0.0
        x = ((pred + 1) / 2 - mean) / std
        y = ((target + 1) / 2 - mean) / std
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 0.4,
        ssim_weight: float = 0.3,
        perceptual_weight: float = 0.2,
        charbonnier_weight: float = 0.1,
        edge_weight: float = 0.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.charbonnier_weight = charbonnier_weight
        self.edge_weight = edge_weight

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None
        self.charbonnier_loss = CharbonnierLoss()

    def _ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = (pred + 1) / 2, (target + 1) / 2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_pred = F.avg_pool2d(pred, 3, 1, 1)
        mu_target = F.avg_pool2d(target, 3, 1, 1)
        sigma_pred = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_target ** 2
        sigma_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target
        ssim_n = (2 * mu_pred * mu_target + C1) * (2 * sigma_pt + C2)
        ssim_d = (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)
        return 1 - ssim_n / (ssim_d + 1e-8)

    def _edge_loss(self, pred, target):
        # Match reference gradients; do not reward sharpness without a target.
        pred, target = (pred + 1) / 2, (target + 1) / 2
        dx = F.l1_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])
        dy = F.l1_loss(pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
        return (dx + dy) / 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l1 = self.l1_loss(pred, target)
        ssim = self._ssim(pred, target).mean()
        perceptual = self.perceptual_loss(pred, target) if self.perceptual_loss is not None else pred.new_zeros(())
        charbonnier = self.charbonnier_loss(pred, target)

        edge = self._edge_loss(pred, target) if self.edge_weight else pred.new_zeros(())

        total = (
            self.l1_weight * l1
            + self.ssim_weight * ssim
            + self.perceptual_weight * perceptual
            + self.charbonnier_weight * charbonnier
            + self.edge_weight * edge
        )

        return {
            "total": total,
            "l1": l1,
            "ssim": ssim,
            "perceptual": perceptual,
            "charbonnier": charbonnier,
            "edge": edge,
        }
