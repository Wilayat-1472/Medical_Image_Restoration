import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return self.sigmoid(attn) * x


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, channels: int, scales: list = [1, 2, 4]):
        super().__init__()
        self.branches = nn.ModuleList()
        for s in scales:
            if s == 1:
                self.branches.append(nn.Identity())
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=s, stride=s),
                        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                    )
                )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = []
        for branch in self.branches:
            out = branch(x)
            if out.shape[-2:] != (h, w):
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(out)
        return self.fusion(torch.cat(outs, dim=1))
