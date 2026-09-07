import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .bayesian import BayesianLinear, _BayesianCausalConv1d, _BayesianParameterLayer, _BayesianTCNResidualBlock
    from .common import MIN_VELOCITY_VARIANCE, VELOCITY_COMPONENTS, TemporalModel
except ImportError:
    from bayesian import BayesianLinear, _BayesianCausalConv1d, _BayesianParameterLayer, _BayesianTCNResidualBlock
    from common import MIN_VELOCITY_VARIANCE, VELOCITY_COMPONENTS, TemporalModel


class UCBTCN(TemporalModel):
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64, tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=.2, ucb_rho=-3., ucb_sig1=0., ucb_sig2=6., ucb_pi=.25, initial_prior_sigma=None, legs=('left', 'right')):
        super().__init__(window_size, num_features, legs)
        kwargs = dict(rho=ucb_rho, sig1=ucb_sig1, sig2=ucb_sig2, pi=ucb_pi, initial_prior_sigma=initial_prior_sigma)
        self.input_proj = _BayesianCausalConv1d(num_features, tcn_num_channels, 1, 1, **kwargs)
        self.tcn_backbone = nn.ModuleList([_BayesianTCNResidualBlock(tcn_num_channels, tcn_kernel_size, 2 ** index, tcn_dropout, **kwargs) for index in range(tcn_num_blocks)])
        self.velocity_head = BayesianLinear(tcn_num_channels, 2 * len(self.legs) * len(VELOCITY_COMPONENTS), **kwargs)
        self.contact_head = BayesianLinear(tcn_num_channels, len(self.legs), **kwargs)

    def extract_features(self, x, sample=False, calculate_log_probs=False):
        features = self.input_proj(x.permute(0, 2, 1), sample, calculate_log_probs)
        for block in self.tcn_backbone: features = block(features, sample, calculate_log_probs)
        return features
    def bayesian_log_probs(self):
        layers = [module for module in self.modules() if isinstance(module, _BayesianParameterLayer)]
        if any(layer.log_prior is None for layer in layers): raise RuntimeError('Call forward(..., calculate_log_probs=True) before requesting UCB log probabilities.')
        return sum(layer.log_prior for layer in layers), sum(layer.log_variational_posterior for layer in layers)
    def kl_to_prior(self): return sum(module.kl_to_prior() for module in self.modules() if isinstance(module, _BayesianParameterLayer))
    @torch.no_grad()
    def snapshot_posterior_as_prior(self):
        for module in self.modules():
            if isinstance(module, _BayesianParameterLayer): module.set_prior_from_posterior()
    set_prior_from_posterior = snapshot_posterior_as_prior
    def forward(self, x, return_sequence=True, sample=False, calculate_log_probs=False, **_):
        features = self.extract_features(x, sample, calculate_log_probs)
        output = self.velocity_head(features.permute(0, 2, 1) if return_sequence else features[:, :, -1], sample, calculate_log_probs).view(*(features.shape[:1] + ((features.shape[-1],) if return_sequence else ())), len(self.legs), 2, len(VELOCITY_COMPONENTS))
        mean, variance = output[..., 0, :], F.softplus(output[..., 1, :]) + MIN_VELOCITY_VARIANCE
        if return_sequence: mean, variance = mean.permute(0, 2, 3, 1), variance.permute(0, 2, 3, 1); sequence = mean, variance
        else: sequence = None, None
        return sequence[0], mean[..., -1] if return_sequence else mean, sequence[1], variance[..., -1] if return_sequence else variance, self.contact_head(features[:, :, -1], sample, calculate_log_probs)
