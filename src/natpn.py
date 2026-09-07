from pathlib import Path
import sys
import torch
import torch.nn as nn

try:
    from .common import MIN_VELOCITY_VARIANCE, VELOCITY_COMPONENTS, contact_logits, make_contact_heads
    from .deterministic import DeterministicTCN
except ImportError:
    from common import MIN_VELOCITY_VARIANCE, VELOCITY_COMPONENTS, contact_logits, make_contact_heads
    from deterministic import DeterministicTCN

_ROOT = Path(__file__).resolve().parents[1]
NATPN_ROOT = next((path for path in (_ROOT / 'NATPN' / 'natural-posterior-network', _ROOT / 'Source Codes' / 'NATPN' / 'natural-posterior-network') if path.is_dir()), None)
if NATPN_ROOT is not None:
    sys.path.insert(0, str(NATPN_ROOT))
    from natpn.nn import BayesianLoss
    from natpn.nn.flow import MaskedAutoregressiveFlow, RadialFlow
    from natpn.nn.output import NormalOutput
    from natpn.nn.scaler import EvidenceScaler
    from natpn.distributions import NormalGammaPrior, PosteriorUpdate
else:
    BayesianLoss = MaskedAutoregressiveFlow = RadialFlow = NormalOutput = EvidenceScaler = NormalGammaPrior = PosteriorUpdate = None


class NatPNVelocityHeads(nn.Module):
    def __init__(self, channels, legs, flow_type='radial', flow_layers=8, certainty_budget='normal'):
        super().__init__()
        if NATPN_ROOT is None:
            raise RuntimeError('NatPN is required for NatPNVelocityHeads; expected NATPN/natural-posterior-network or Source Codes/NATPN/natural-posterior-network.')
        if flow_type not in ('radial', 'masked_autoregressive'):
            raise ValueError("natpn_flow_type must be 'radial' or 'masked_autoregressive'.")
        flow = RadialFlow(channels, flow_layers) if flow_type == 'radial' else MaskedAutoregressiveFlow(channels, num_layers=flow_layers)
        self.legs, self.flow, self.scaler = tuple(legs), flow, EvidenceScaler(channels, certainty_budget)
        self.outputs = nn.ModuleDict({leg: nn.ModuleList([NormalOutput(channels) for _ in VELOCITY_COMPONENTS]) for leg in self.legs})
        for outputs in self.outputs.values():
            for output in outputs:
                output.prior = NormalGammaPrior(mean=0., lambd=1e-6, alpha=2, beta=1.)

    def flow_nll(self, features, return_sequence):
        inputs = features.permute(0, 2, 1).reshape(-1, features.shape[1]) if return_sequence else features[:, :, -1]
        return -self.flow(inputs.detach()).mean()

    @staticmethod
    def _variance(posterior):
        aleatoric = posterior.beta / (posterior.alpha - 1.).clamp_min(MIN_VELOCITY_VARIANCE)
        return aleatoric + aleatoric / posterior.lambd

    def forward(self, features, return_sequence):
        batch, _, steps = features.shape
        inputs = features.permute(0, 2, 1).reshape(-1, features.shape[1]) if return_sequence else features[:, :, -1]
        evidence = self.scaler(self.flow(inputs))
        posteriors = {leg: [output.prior.update(PosteriorUpdate(output(inputs).expected_sufficient_statistics(), evidence)) for output in self.outputs[leg]] for leg in self.legs}
        mean = torch.stack([torch.stack([posterior.maximum_a_posteriori().mean() for posterior in posteriors[leg]], -1) for leg in self.legs], 1)
        variance = torch.stack([torch.stack([self._variance(posterior) for posterior in posteriors[leg]], -1) for leg in self.legs], 1)
        if not return_sequence:
            return None, mean, None, variance, tuple(tuple(posteriors[leg]) for leg in self.legs)
        mean = mean.view(batch, steps, len(self.legs), 3).permute(0, 2, 3, 1)
        variance = variance.view(batch, steps, len(self.legs), 3).permute(0, 2, 3, 1)
        shaped = tuple(tuple(type(p)(p.mu.view(batch, steps), p.lambd.view(batch, steps), p.alpha.view(batch, steps), p.beta.view(batch, steps)) for p in posteriors[leg]) for leg in self.legs)
        return mean, mean[..., -1], variance, variance[..., -1], shaped


class TCN(DeterministicTCN):
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64, tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=.2, natpn_flow_type='radial', natpn_flow_layers=8, natpn_certainty_budget='normal', legs=('left', 'right')):
        super().__init__(window_size, num_features, tcn_num_channels, tcn_kernel_size, tcn_num_blocks, tcn_dropout, legs)
        self.velocity_heads = NatPNVelocityHeads(tcn_num_channels, self.legs, natpn_flow_type, natpn_flow_layers, natpn_certainty_budget)
        self.contact_heads = make_contact_heads(tcn_num_channels, self.legs)

    def forward(self, x, return_sequence=True, return_posteriors=False, return_flow_nll=False):
        features = self.extract_features(x)
        if return_flow_nll:
            return self.velocity_heads.flow_nll(features, return_sequence)
        velocity = self.velocity_heads(features, return_sequence)
        outputs = (*velocity[:4], contact_logits(self.contact_heads, features, self.legs))
        return (*outputs, velocity[4]) if return_posteriors else outputs
