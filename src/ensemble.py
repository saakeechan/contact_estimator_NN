import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import MIN_VELOCITY_VARIANCE, VELOCITY_COMPONENTS, contact_logits, make_contact_heads
    from .deterministic import DeterministicTCN
except ImportError:
    from common import MIN_VELOCITY_VARIANCE, VELOCITY_COMPONENTS, contact_logits, make_contact_heads
    from deterministic import DeterministicTCN


def make_velocity_heads(channels, legs):
    return nn.ModuleDict({leg: nn.Sequential(nn.Linear(channels, 512), nn.SiLU(), nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 2 * len(VELOCITY_COMPONENTS))) for leg in legs})


def predict_velocity(heads, features, legs, return_sequence=True):
    inputs = features.permute(0, 2, 1) if return_sequence else features[:, :, -1]
    outputs = [heads[leg](inputs) for leg in legs]
    means = torch.stack([output[..., :3] for output in outputs], dim=1)
    variances = torch.stack([F.softplus(output[..., 3:]) + MIN_VELOCITY_VARIANCE for output in outputs], dim=1)
    if not return_sequence:
        return None, means, None, variances
    means, variances = means.permute(0, 1, 3, 2), variances.permute(0, 1, 3, 2)
    return means, means[..., -1], variances, variances[..., -1]


class EnsembleTCN(DeterministicTCN):
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64, tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=.2, legs=('left', 'right')):
        super().__init__(window_size, num_features, tcn_num_channels, tcn_kernel_size, tcn_num_blocks, tcn_dropout, legs)
        self.velocity_heads, self.contact_heads = make_velocity_heads(tcn_num_channels, self.legs), make_contact_heads(tcn_num_channels, self.legs)

    def forward(self, x, return_sequence=True, **_):
        features = self.extract_features(x)
        return (*predict_velocity(self.velocity_heads, features, self.legs, return_sequence), contact_logits(self.contact_heads, features, self.legs))
