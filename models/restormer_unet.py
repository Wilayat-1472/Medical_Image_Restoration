import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseRestorationModel


class MDTA(nn.Module):
    def __init__(self, channels: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.view(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    def __init__(self, channels: int, expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, kind="WithBias"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels)) if kind == "WithBias" else None

    def forward(self, x):
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        if self.bias is not None:
            x = x - x.mean(dim=1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-5)
        x = x * self.weight[None, :, None, None]
        if self.bias is not None:
            x = x + self.bias[None, :, None, None]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, ffn_expansion=2.66, bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.norm1 = LayerNorm2d(channels, LayerNorm_type)
        self.attn = MDTA(channels, num_heads, bias)
        self.norm2 = LayerNorm2d(channels, LayerNorm_type)
        self.ffn = GDFN(channels, ffn_expansion, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class RestormerEncoder(nn.Module):
    def __init__(self, channels: int, num_blocks: list, num_heads: list, ffn_expansion: float, bias: bool, LayerNorm_type: str):
        super().__init__()
        self.levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = channels
        for i, (nb, nh) in enumerate(zip(num_blocks, num_heads)):
            level = nn.Sequential(*[TransformerBlock(ch, nh, ffn_expansion, bias, LayerNorm_type) for _ in range(nb)])
            self.levels.append(level)
            if i < len(num_blocks) - 1:
                self.downsamplers.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
                ch *= 2

    def forward(self, x: torch.Tensor) -> list:
        skips = []
        for i, level in enumerate(self.levels):
            x = level(x)
            skips.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        return x, skips


class RestormerDecoder(nn.Module):
    def __init__(self, channels: int, num_blocks: list, num_heads: list, ffn_expansion: float, bias: bool, LayerNorm_type: str):
        super().__init__()
        self.levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        ch = channels * (2 ** (len(num_blocks) - 1))
        for i, (nb, nh) in enumerate(zip(reversed(num_blocks), reversed(num_heads))):
            level = nn.Sequential(*[TransformerBlock(ch, nh, ffn_expansion, bias, LayerNorm_type) for _ in range(nb)])
            self.levels.append(level)
            if i < len(num_blocks) - 1:
                up_ch = ch // 2
                self.upsamplers.append(nn.ConvTranspose2d(ch, up_ch, kernel_size=2, stride=2))
                ch = up_ch

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        for i, (level, up) in enumerate(zip(self.levels, list(self.upsamplers) + [None])):
            x = level(x)
            if up is not None:
                x = up(x)
                skip = skips[-(i + 2)]
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = x + skip
        return x


class RestormerUNet(BaseRestorationModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list = [4, 6, 6, 8],
        num_heads: list = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        self.encoder = RestormerEncoder(dim, num_blocks, num_heads, ffn_expansion_factor, bias, LayerNorm_type)

        bottleneck_ch = dim * (2 ** (len(num_blocks) - 1))
        self.bottleneck = nn.Sequential(*[
            TransformerBlock(bottleneck_ch, num_heads[-1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(2)
        ])

        self.decoder = RestormerDecoder(dim, num_blocks, num_heads, ffn_expansion_factor, bias, LayerNorm_type)

        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.Tanh(),
        )

        self._name = "restormer_unet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        stride = 2 ** (len(self.encoder.levels) - 1)
        x = F.pad(x, (0, (-w) % stride, 0, (-h) % stride), mode="replicate")
        x = self.input_conv(x)
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output_conv(x)[..., :h, :w]
