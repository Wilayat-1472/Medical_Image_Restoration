import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv_mean = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        mean = self.conv_mean(x)
        logvar = self.conv_logvar(x)
        logvar = torch.clamp(logvar, -10, 10)
        return mean, logvar


def compute_confidence_map(logvar: torch.Tensor) -> torch.Tensor:
    variance = torch.exp(logvar)
    max_var = variance.max(dim=1, keepdim=True)[0]
    confidence = torch.exp(-variance / (max_var + 1e-8))
    return confidence


def compute_confidence_score(confidence_map: torch.Tensor) -> torch.Tensor:
    b, c, h, w = confidence_map.shape
    score = confidence_map.view(b, -1).mean(dim=1)
    return score
