import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import MIN_VELOCITY_VARIANCE, contact_logits, make_contact_heads
    from .deterministic import DeterministicTCN
except ImportError:
    from common import MIN_VELOCITY_VARIANCE, contact_logits, make_contact_heads
    from deterministic import DeterministicTCN


class DERVelocityHeads(nn.Module):
    def __init__(self, channels, legs):
        super().__init__()
        self.legs = tuple(legs)
        self.heads = nn.ModuleDict({leg: nn.Sequential(nn.Linear(channels, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 12)) for leg in self.legs})

    @staticmethod
    def _distribution_parameters(output):
        gamma, nu, alpha, beta = output.chunk(4, dim=-1)
        return gamma, F.softplus(nu) + MIN_VELOCITY_VARIANCE, F.softplus(alpha) + 1., F.softplus(beta) + MIN_VELOCITY_VARIANCE

    def forward(self, features, return_sequence):
        inputs = features.permute(0, 2, 1) if return_sequence else features[:, :, -1]
        params = [torch.stack(values, dim=1) for values in zip(*[self._distribution_parameters(self.heads[leg](inputs)) for leg in self.legs])]
        gamma, nu, alpha, beta = params
        variance = beta * (1. + nu) / (nu * (alpha - 1.).clamp_min(MIN_VELOCITY_VARIANCE))
        if not return_sequence:
            return None, gamma, None, variance, tuple(params)
        gamma, variance = gamma.permute(0, 1, 3, 2), variance.permute(0, 1, 3, 2)
        params = tuple(value.permute(0, 1, 3, 2) for value in params)
        return gamma, gamma[..., -1], variance, variance[..., -1], params


class DERTCN(DeterministicTCN):
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64, tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=.2, legs=('left', 'right')):
        super().__init__(window_size, num_features, tcn_num_channels, tcn_kernel_size, tcn_num_blocks, tcn_dropout, legs)
        self.velocity_heads, self.contact_heads = DERVelocityHeads(tcn_num_channels, self.legs), make_contact_heads(tcn_num_channels, self.legs)

    def forward(self, x, return_sequence=True, return_posteriors=False, **_):
        features = self.extract_features(x)
        velocity = self.velocity_heads(features, return_sequence)
        outputs = (*velocity[:4], contact_logits(self.contact_heads, features, self.legs))
        return (*outputs, velocity[4]) if return_posteriors else outputs
