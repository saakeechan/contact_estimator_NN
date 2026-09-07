import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import TemporalModel
except ImportError:
    from common import TemporalModel


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.dropout1(F.silu(self.conv1(x)))
        out = self.dropout2(F.silu(self.conv2(out)))
        residual = x if self.downsample is None else self.downsample(x)
        return F.silu(out + residual)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class DeterministicTCN(TemporalModel):
    """Shared deterministic temporal encoder used by Gaussian, DER, and NatPN."""
    def __init__(self, window_size, num_features, channels, kernel_size, blocks, dropout, legs):
        super().__init__(window_size, num_features, legs)
        self.input_proj = nn.Conv1d(num_features, channels, kernel_size=1)
        self.tcn_backbone = nn.Sequential(*[TCNResidualBlock(channels, channels, kernel_size, 2 ** index, dropout) for index in range(blocks)])

    def extract_features(self, x):
        return self.tcn_backbone(self.input_proj(x.permute(0, 2, 1)))
