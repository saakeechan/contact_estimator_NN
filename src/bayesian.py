import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UCB_MIN_RHO, UCB_MAX_RHO = -12., 5.


class _UCBPrior(nn.Module):
    def __init__(self, sig1=0., sig2=6., pi=.25):
        super().__init__()
        if not 0. < pi < 1.: raise ValueError('ucb_pi must be strictly between zero and one.')
        self.sig1, self.sig2, self.pi = sig1, sig2, pi
    def log_prob(self, value):
        first, second = torch.distributions.Normal(0., value.new_tensor(np.exp(-self.sig1))).log_prob(value), torch.distributions.Normal(0., value.new_tensor(np.exp(-self.sig2))).log_prob(value)
        return torch.logaddexp(first + np.log(self.pi), second + np.log1p(-self.pi)).sum()


class _BayesianParameterLayer(nn.Module):
    def __init__(self, rho=-3., sig1=0., sig2=6., pi=.25, initial_prior_sigma=None):
        super().__init__(); self.rho, self.prior, self.initial_prior_sigma = rho, _UCBPrior(sig1, sig2, pi), initial_prior_sigma; self.log_prior = self.log_variational_posterior = None
    @staticmethod
    def posterior_scale(rho):
        return F.softplus(torch.nan_to_num(rho, nan=-3., posinf=UCB_MAX_RHO, neginf=UCB_MIN_RHO).clamp(UCB_MIN_RHO, UCB_MAX_RHO)).clamp_min(1e-6)
    def _initialize_continual_prior(self):
        def initial(mu, rho): return (mu.detach().clone(), self.posterior_scale(rho).detach().clone()) if self.initial_prior_sigma is None else (torch.zeros_like(mu), torch.full_like(mu, float(self.initial_prior_sigma)).clamp_min(1e-8))
        weight_mu, weight_sigma = initial(self.weight_mu, self.weight_rho)
        self.register_buffer('prior_weight_mu', weight_mu); self.register_buffer('prior_weight_sigma', weight_sigma)
        if self.bias_mu is None: self.register_buffer('prior_bias_mu', None); self.register_buffer('prior_bias_sigma', None)
        else:
            bias_mu, bias_sigma = initial(self.bias_mu, self.bias_rho); self.register_buffer('prior_bias_mu', bias_mu); self.register_buffer('prior_bias_sigma', bias_sigma)
        self.register_buffer('uses_previous_posterior_prior', torch.tensor(self.initial_prior_sigma is not None))
    def posterior_sigma(self): return self.posterior_scale(self.weight_rho)
    def kl_to_prior(self):
        def kl(mu, rho, prior_mu, prior_sigma):
            sigma, prior_sigma = self.posterior_scale(rho), prior_sigma.clamp_min(1e-8)
            return (torch.log(prior_sigma / sigma) + (sigma.square() + (mu - prior_mu).square()) / (2 * prior_sigma.square()) - .5).sum()
        return kl(self.weight_mu, self.weight_rho, self.prior_weight_mu, self.prior_weight_sigma) + (0 if self.bias_mu is None else kl(self.bias_mu, self.bias_rho, self.prior_bias_mu, self.prior_bias_sigma))
    @torch.no_grad()
    def set_prior_from_posterior(self):
        self.prior_weight_mu.copy_(self.weight_mu.detach()); self.prior_weight_sigma.copy_(self.posterior_scale(self.weight_rho.detach()))
        if self.bias_mu is not None: self.prior_bias_mu.copy_(self.bias_mu.detach()); self.prior_bias_sigma.copy_(self.posterior_scale(self.bias_rho.detach()))
        self.uses_previous_posterior_prior.fill_(True)
    snapshot_posterior_as_prior = set_prior_from_posterior
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for current, legacy in (('prior_weight_mu','weight_prior_mu'), ('prior_weight_sigma','weight_prior_sigma'), ('prior_bias_mu','bias_prior_mu'), ('prior_bias_sigma','bias_prior_sigma')):
            if prefix + current not in state_dict and prefix + legacy in state_dict: state_dict[prefix + current] = state_dict[prefix + legacy]
            state_dict.pop(prefix + legacy, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    def _sample(self, mu, rho, prior_mu, prior_sigma, sample, calculate):
        value = mu + self.posterior_scale(rho) * torch.randn_like(mu) if self.training or sample else mu
        if not (self.training or calculate): return value, value.new_zeros(()), value.new_zeros(())
        prior = torch.distributions.Normal(prior_mu, prior_sigma).log_prob(value).sum() if self.uses_previous_posterior_prior else self.prior.log_prob(value)
        return value, prior, torch.distributions.Normal(mu, self.posterior_scale(rho)).log_prob(value).sum()


class _BayesianAffine(_BayesianParameterLayer):
    def __init__(self, weight_shape, bias_shape, **kwargs):
        super().__init__(**kwargs); self.weight_mu = nn.Parameter(torch.empty(weight_shape).normal_(0., .1)); self.weight_rho = nn.Parameter(torch.empty_like(self.weight_mu).normal_(self.rho, .1)); self.bias_mu = nn.Parameter(torch.empty(bias_shape).normal_(0., .1)) if bias_shape else None; self.bias_rho = nn.Parameter(torch.empty_like(self.bias_mu).normal_(self.rho, .1)) if bias_shape else None; self._initialize_continual_prior()
    def parameters_for_forward(self, sample, calculate):
        weight, prior, posterior = self._sample(self.weight_mu, self.weight_rho, self.prior_weight_mu, self.prior_weight_sigma, sample, calculate)
        if self.bias_mu is None: bias = None
        else:
            bias, bp, bq = self._sample(self.bias_mu, self.bias_rho, self.prior_bias_mu, self.prior_bias_sigma, sample, calculate); prior, posterior = prior + bp, posterior + bq
        self.log_prior, self.log_variational_posterior = prior, posterior; return weight, bias


class BayesianConv1d(_BayesianAffine):
    def __init__(self, in_channels, out_channels, kernel_size, *, dilation=1, bias=True, **kwargs): super().__init__((out_channels, in_channels, kernel_size), out_channels if bias else None, **kwargs); self.dilation = dilation
    def forward(self, x, sample=False, calculate_log_probs=False):
        weight, bias = self.parameters_for_forward(sample, calculate_log_probs); return F.conv1d(x, weight, bias, dilation=self.dilation)


class BayesianLinear(_BayesianAffine):
    def __init__(self, in_features, out_features, *, bias=True, **kwargs): super().__init__((out_features, in_features), out_features if bias else None, **kwargs)
    def forward(self, x, sample=False, calculate_log_probs=False):
        weight, bias = self.parameters_for_forward(sample, calculate_log_probs); return F.linear(x, weight, bias)


class _BayesianCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs): super().__init__(); self.pad = (kernel_size - 1) * dilation; self.conv = BayesianConv1d(in_channels, out_channels, kernel_size, dilation=dilation, **kwargs)
    def forward(self, x, sample=False, calculate_log_probs=False): return self.conv(F.pad(x, (self.pad, 0)), sample, calculate_log_probs)


class _BayesianTCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout, **kwargs): super().__init__(); self.conv1 = _BayesianCausalConv1d(channels, channels, kernel_size, dilation, **kwargs); self.conv2 = _BayesianCausalConv1d(channels, channels, kernel_size, dilation, **kwargs); self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
    def forward(self, x, sample=False, calculate_log_probs=False): return F.silu(self.dropout2(F.silu(self.conv2(self.dropout1(F.silu(self.conv1(x, sample, calculate_log_probs))), sample, calculate_log_probs))) + x)
