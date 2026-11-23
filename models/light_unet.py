import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseRestorationModel
from .attention import ChannelAttention


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class LightEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        self.attention = ChannelAttention(out_ch) if use_se else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = x + identity
        skip = x
        x = self.pool(x)
        return x, skip


class LightDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = DepthwiseSeparableConv(out_ch * 2, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.attention = ChannelAttention(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x


class LightUNet(BaseRestorationModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 32,
        depth: int = 4,
        use_se: bool = True,
    ):
        super().__init__()
        self.input_conv = DepthwiseSeparableConv(in_channels, base_filters)

        encoders = []
        ch = base_filters
        for i in range(depth):
            out_ch = base_filters * (2 ** min(i, 3))
            encoders.append(LightEncoderBlock(ch, out_ch, use_se=use_se))
            ch = out_ch
        self.encoders = nn.ModuleList(encoders)

        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(ch, ch * 2),
            DepthwiseSeparableConv(ch * 2, ch),
        )

        decoders = []
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** min(i, 3))
            decoders.append(LightDecoderBlock(ch, out_ch, use_se=use_se))
            ch = out_ch
        self.decoders = nn.ModuleList(decoders)

        self.output_conv = nn.Sequential(
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        )

        self._name = "light_unet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder in self.decoders:
            skip = skips.pop()
            x = decoder(x, skip)

        return self.output_conv(x)
