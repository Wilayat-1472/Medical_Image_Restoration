"""Residual Attention U-Net architecture matching the trained LDCT joblib bundle."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double 3x3 convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive soft-attention gate for skip connections."""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.wg = nn.Conv2d(f_g, f_int, kernel_size=1, bias=False)
        self.wx = nn.Conv2d(f_l, f_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(f_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.wg(g) + self.wx(x))
        return x * alpha


class ResidualAttentionUNet(nn.Module):
    """4-level Residual Attention U-Net predicting noise residual to subtract from LDCT."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_c: int = 32):
        super().__init__()
        f1, f2, f3, f4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.enc1 = ConvBlock(in_channels, f1)
        self.enc2 = ConvBlock(f1, f2)
        self.enc3 = ConvBlock(f2, f3)
        self.bot = ConvBlock(f3, f4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(f3, f3, f3 // 2)
        self.dec3 = ConvBlock(f3 * 2, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(f2, f2, f2 // 2)
        self.dec2 = ConvBlock(f2 * 2, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(f1, f1, f1 // 2)
        self.dec1 = ConvBlock(f1 * 2, f1)

        self.noise_head = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        inp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x

        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))

        g3 = self.up3(b)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))

        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))

        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))

        out = torch.clamp(inp - self.noise_head(d1), 0.0, 1.0)
        return out[..., :h, :w]
