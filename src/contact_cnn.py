import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

import numpy as np

VELOCITY_COMPONENTS = ("x", "y", "z")
LEGS = ("left",)
MIN_VELOCITY_VARIANCE = 1e-6
UCB_MIN_RHO = -12.0
UCB_MAX_RHO = 5.0

NATPN_ROOT = Path(__file__).resolve().parents[1] / "NATPN" / "natural-posterior-network"
if not NATPN_ROOT.is_dir():
    raise RuntimeError(f"Bundled NatPN source is missing: {NATPN_ROOT}")
if str(NATPN_ROOT) not in sys.path:
    sys.path.insert(0, str(NATPN_ROOT))

from natpn.nn.flow import MaskedAutoregressiveFlow, RadialFlow
from natpn.nn.output import NormalOutput
from natpn.nn.scaler import EvidenceScaler
from natpn.distributions import PosteriorUpdate


def make_velocity_heads(num_features):
    """Predict body-velocity mean and diagonal variance for [vx, vy, vz]."""
    return nn.ModuleDict({
        leg: nn.Sequential(
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 2 * len(VELOCITY_COMPONENTS)),
        )
        for leg in LEGS
    })


def predict_velocity(velocity_heads, features, return_sequence=True):
    """Predict all timesteps for dense supervision, or only the final timestep."""
    if not return_sequence:
        outputs = [velocity_heads[leg](features[:, :, -1]) for leg in LEGS]  # [B, 6]
        velocity_out = torch.stack([output[..., :3] for output in outputs], dim=1)
        covariance_out = torch.stack([
            F.softplus(output[..., 3:]) + MIN_VELOCITY_VARIANCE for output in outputs
        ], dim=1)
        return None, velocity_out, None, covariance_out

    features_by_time = features.permute(0, 2, 1)  # [B, T, channels]
    outputs = [velocity_heads[leg](features_by_time) for leg in LEGS]  # [B, T, 6]
    velocity_seq = torch.stack([output[..., :3].permute(0, 2, 1) for output in outputs], dim=1)
    covariance_seq = torch.stack([
        (F.softplus(output[..., 3:]) + MIN_VELOCITY_VARIANCE).permute(0, 2, 1)
        for output in outputs
    ], dim=1)
    return velocity_seq, velocity_seq[:, :, :, -1], covariance_seq, covariance_seq[:, :, :, -1]


def make_contact_heads(num_features):
    return nn.ModuleDict({
        leg: nn.Sequential(
            nn.Linear(num_features, 256),
            nn.SiLU(),
            nn.Linear(256, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        for leg in LEGS
    })


class NatPNVelocityHeads(nn.Module):
    """Task-latent NatPN heads with one shared flow-derived evidence space."""
    def __init__(self, latent_dim, flow_type="radial", flow_layers=8, certainty_budget="normal"):
        super().__init__()
        if flow_type == "radial":
            flow = lambda: RadialFlow(latent_dim, flow_layers)
        elif flow_type == "masked_autoregressive":
            flow = lambda: MaskedAutoregressiveFlow(latent_dim, num_layers=flow_layers)
        else:
            raise ValueError("natpn_flow_type must be 'radial' or 'masked_autoregressive'.")
        self.flow = flow()
        self.scaler = EvidenceScaler(latent_dim, certainty_budget)
        self.outputs = nn.ModuleList([NormalOutput(latent_dim) for _ in VELOCITY_COMPONENTS])

    @staticmethod
    def predictive_variance(posterior):
        aleatoric = posterior.beta / (posterior.alpha - 1.0).clamp_min(MIN_VELOCITY_VARIANCE)
        return aleatoric + aleatoric / posterior.lambd

    def flow_nll(self, features, return_sequence):
        """Fit the task-latent flows."""
        inputs = features.permute(0, 2, 1).reshape(-1, features.shape[1]) if return_sequence else features[:, :, -1]
        return -self.flow(inputs.detach()).mean()

    def forward(self, features, return_sequence):
        if return_sequence:
            batch_size, _, num_steps = features.shape
            inputs = features.permute(0, 2, 1).reshape(-1, features.shape[1])
        else:
            batch_size, _, num_steps = features.shape
            inputs = features[:, :, -1]

        log_evidence = self.scaler(self.flow(inputs))
        evidence = log_evidence.exp()
        likelihoods = [output(inputs) for output in self.outputs]
        posteriors = [
            output.prior.update(
                PosteriorUpdate(likelihood.expected_sufficient_statistics(), log_evidence)
            )
            for output, likelihood in zip(self.outputs, likelihoods)
        ]
        means = torch.stack([posterior.maximum_a_posteriori().mean() for posterior in posteriors], dim=-1)
        variances = torch.stack([self.predictive_variance(posterior) for posterior in posteriors], dim=-1)

        if return_sequence:
            means = means.view(batch_size, num_steps, len(VELOCITY_COMPONENTS)).permute(0, 2, 1).unsqueeze(1)
            variances = variances.view(batch_size, num_steps, len(VELOCITY_COMPONENTS)).permute(0, 2, 1).unsqueeze(1)
            posteriors = [
                type(posterior)(
                    posterior.mu.view(batch_size, num_steps),
                    posterior.lambd.view(batch_size, num_steps),
                    posterior.alpha.view(batch_size, num_steps),
                    posterior.beta.view(batch_size, num_steps),
                )
                for posterior in posteriors
            ]
            return means, means[:, :, :, -1], variances, variances[:, :, :, -1], posteriors

        return None, means.unsqueeze(1), None, variances.unsqueeze(1), posteriors


class DERVelocityHeads(nn.Module):
    """Normal-Inverse-Gamma velocity head for deep evidential regression."""
    def __init__(self, latent_dim):
        super().__init__()
        self.heads = nn.ModuleDict({
            leg: nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.SiLU(),
                nn.Linear(128,128),
                nn.SiLU(),
                nn.Linear(128, 4 * len(VELOCITY_COMPONENTS)),
            ) for leg in LEGS
        })

    @staticmethod
    def _nig_parameters(output):
        gamma, raw_nu, raw_alpha, raw_beta = output.chunk(4, dim=-1)
        return gamma, F.softplus(raw_nu) + MIN_VELOCITY_VARIANCE, F.softplus(raw_alpha) + 1.0, F.softplus(raw_beta) + MIN_VELOCITY_VARIANCE

    @staticmethod
    def _variance(nu, alpha, beta):
        return beta * (1.0 + nu) / (nu * (alpha - 1.0).clamp_min(MIN_VELOCITY_VARIANCE))

    def forward(self, features, return_sequence):
        if return_sequence:
            outputs = [self.heads[leg](features.permute(0, 2, 1)) for leg in LEGS]
            params = [torch.stack(values, dim=1).permute(0, 1, 3, 2) for values in zip(*[self._nig_parameters(output) for output in outputs])]
        else:
            outputs = [self.heads[leg](features[:, :, -1]) for leg in LEGS]
            params = [torch.stack(values, dim=1) for values in zip(*[self._nig_parameters(output) for output in outputs])]
        gamma, nu, alpha, beta = params
        variance = self._variance(nu, alpha, beta)
        if return_sequence:
            return gamma, gamma[:, :, :, -1], variance, variance[:, :, :, -1], tuple(params)
        return None, gamma, None, variance, tuple(params)

"""
Two network architectures are available:

1. TCN (Pure Temporal Convolutional Network):
   - Residual blocks with exponentially increasing dilation
   - Weight normalization (not batch norm)
   - Exponentially increasing dilation rates: 2^0, 2^1, 2^2, ...
   - Good balance between expressiveness and speed
   - Configurable: tcn_num_channels, tcn_kernel_size, tcn_num_blocks, tcn_dropout

2. contact_cnn (Vanilla CNN):
   - Simple sequential convolutional architecture
   - Three conv layers with kernel size 3
   - Uses SiLU activation
   - No residual connections
   - Fastest, fewest parameters

All architectures:
- Support causal convolutions (no future information leakage)
- Output body-velocity predictions for all timesteps (dense supervision)
- Extract last timestep for inference
"""


class TCNResidualBlock(nn.Module):
    """
    Residual block for TCN as shown in Figure 2.
    Contains two causal convolution layers with weight normalization, dropout, and skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        
        # First causal conv layer (with weight normalization)
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.silu1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second causal conv layer (with weight normalization)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.silu2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 1x1 convolution for residual connection if dimensions don't match
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.silu_out = nn.SiLU()
        
    def forward(self, x):
        # Save input for residual connection
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.silu1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.silu2(out)
        out = self.dropout2(out)
        
        # Apply 1x1 conv to residual if needed
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Add residual connection
        out = out + residual
        out = self.silu_out(out)
        
        return out


class TCN(nn.Module):
    """
    Pure Temporal Convolutional Network (TCN).
    Uses residual blocks with exponentially increasing dilation rates.
    
    Architecture:
    1. Input projection: map raw features to tcn_num_channels dimension
    2. TCN backbone: stacked residual blocks with increasing dilation
    3. Velocity prediction heads
    
    Key design choices:
    - Exponentially increasing dilation: 2^0, 2^1, 2^2, ... (receptive field grows exponentially)
    - Residual connections: enable training of deep networks
    - Weight normalization: stabilizes training without batch statistics
    - Causal convolutions: no future information leakage
    """
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64,
                 tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=0.2,
                 natpn_flow_type="radial", natpn_flow_layers=8, natpn_certainty_budget="normal"):
        super(TCN, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        # 1. Project raw features into TCN channel dimension
        self.input_proj = nn.Conv1d(num_features, tcn_num_channels, kernel_size=1)
        
        # 2. TCN backbone with residual blocks
        tcn_layers = []
        for i in range(tcn_num_blocks):
            dilation_rate = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, ...
            
            tcn_layers.append(TCNResidualBlock(
                tcn_num_channels, tcn_num_channels, 
                tcn_kernel_size, dilation_rate, tcn_dropout
            ))
        
        self.tcn_backbone = nn.Sequential(*tcn_layers)
        
        self.velocity_heads = NatPNVelocityHeads(
            tcn_num_channels, natpn_flow_type, natpn_flow_layers, natpn_certainty_budget
        )
        
        # 4. Left-foot contact logit.
        self.contact_heads = make_contact_heads(tcn_num_channels)
    
    def forward(self, x, return_sequence=True, return_posteriors=False, return_flow_nll=False):
        """
        Args:
            x: (batch_size, window_size, num_features) - RAW features
        
        Returns:
            velocity_seq: (batch_size, 1, 3, window_size) - body-frame body velocities
            velocity_out: (batch_size, 1, 3) - body-frame last-timestep velocity
            covariance_seq: (batch_size, 1, 3, window_size) - diagonal body-velocity variances
            covariance_out: (batch_size, 1, 3) - diagonal last-timestep body-velocity variances
            contact_out: (batch_size, 1) - left-foot contact logit at last timestep
        """
        # 1. Convert to Conv1d format
        x = x.permute(0, 2, 1)  # [B, T, F] → [B, F, T]
        
        # 2. Project to TCN channel dimension
        z = self.input_proj(x)  # [B, F, T] → [B, tcn_num_channels, T]
        
        # 3. TCN backbone
        features = self.tcn_backbone(z)  # [B, tcn_num_channels, T]
        if return_flow_nll:
            return self.velocity_heads.flow_nll(features, return_sequence)
        
        # 4. Velocity prediction
        velocity_seq, velocity_out, covariance_seq, covariance_out, posteriors = self.velocity_heads(
            features, return_sequence,
        )
        
        # 5. Contact prediction (MLP on last timestep features)
        features_last = features[:, :, -1]  # [B, tcn_num_channels]
        contact_out = torch.cat([self.contact_heads[leg](features_last) for leg in LEGS], dim=1)
        
        outputs = velocity_seq, velocity_out, covariance_seq, covariance_out, contact_out
        return (*outputs, posteriors) if return_posteriors else outputs


class DERTCN(nn.Module):
    """TCN with a Normal-Inverse-Gamma body-velocity output head."""
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64,
                 tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.input_proj = nn.Conv1d(num_features, tcn_num_channels, kernel_size=1)
        self.tcn_backbone = nn.Sequential(*[
            TCNResidualBlock(tcn_num_channels, tcn_num_channels, tcn_kernel_size, 2 ** i, tcn_dropout)
            for i in range(tcn_num_blocks)
        ])
        self.velocity_heads = DERVelocityHeads(tcn_num_channels)
        self.contact_heads = make_contact_heads(tcn_num_channels)

    def forward(self, x, return_sequence=True, return_posteriors=False, **_):
        features = self.tcn_backbone(self.input_proj(x.permute(0, 2, 1)))
        velocity_seq, velocity_out, covariance_seq, covariance_out, nig_params = self.velocity_heads(
            features, return_sequence
        )
        contact_out = torch.cat([self.contact_heads[leg](features[:, :, -1]) for leg in LEGS], dim=1)
        outputs = velocity_seq, velocity_out, covariance_seq, covariance_out, contact_out
        return (*outputs, nig_params) if return_posteriors else outputs


class EnsembleTCN(nn.Module):
    """TCN member for a deep ensemble with Gaussian velocity likelihoods."""
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64,
                 tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.input_proj = nn.Conv1d(num_features, tcn_num_channels, kernel_size=1)
        self.tcn_backbone = nn.Sequential(*[
            TCNResidualBlock(tcn_num_channels, tcn_num_channels, tcn_kernel_size, 2 ** i, tcn_dropout)
            for i in range(tcn_num_blocks)
        ])
        self.velocity_heads = make_velocity_heads(tcn_num_channels)
        self.contact_heads = make_contact_heads(tcn_num_channels)

    def forward(self, x, return_sequence=True, **_):
        features = self.tcn_backbone(self.input_proj(x.permute(0, 2, 1)))
        velocity_seq, velocity_out, covariance_seq, covariance_out = predict_velocity(
            self.velocity_heads, features, return_sequence
        )
        contact_out = torch.cat([self.contact_heads[leg](features[:, :, -1]) for leg in LEGS], dim=1)
        return velocity_seq, velocity_out, covariance_seq, covariance_out, contact_out


class MCDropoutTCN(EnsembleTCN):
    """Gaussian-likelihood TCN evaluated through repeated dropout passes."""
    pass


class _UCBPrior(nn.Module):
    """Scale-mixture Gaussian prior used by the bundled UCB implementation."""
    def __init__(self, sig1=0.0, sig2=6.0, pi=0.25):
        super().__init__()
        if not 0.0 < pi < 1.0:
            raise ValueError("ucb_pi must be strictly between zero and one.")
        self.sig1, self.sig2, self.pi = sig1, sig2, pi

    def log_prob(self, value):
        # UCB/src parameterizes its standard deviations as exp(-sig).
        scale1 = value.new_tensor(np.exp(-self.sig1))
        scale2 = value.new_tensor(np.exp(-self.sig2))
        log_prob1 = torch.distributions.Normal(0.0, scale1).log_prob(value)
        log_prob2 = torch.distributions.Normal(0.0, scale2).log_prob(value)
        return torch.logaddexp(log_prob1 + np.log(self.pi), log_prob2 + np.log1p(-self.pi)).sum()


class _BayesianParameterLayer(nn.Module):
    """Shared BBB sampling plus an optional frozen previous-task Gaussian prior."""
    def __init__(self, rho=-3.0, sig1=0.0, sig2=6.0, pi=0.25):
        super().__init__()
        self.rho = rho
        self.prior = _UCBPrior(sig1, sig2, pi)
        self.log_prior = None
        self.log_variational_posterior = None

    def _initialize_continual_prior(self):
        """Create checkpointed buffers; task 0 still uses the mixture prior."""
        self.register_buffer("weight_prior_mu", self.weight_mu.detach().clone())
        self.register_buffer("weight_prior_sigma", self.posterior_scale(self.weight_rho).detach().clone())
        if self.bias_mu is not None:
            self.register_buffer("bias_prior_mu", self.bias_mu.detach().clone())
            self.register_buffer("bias_prior_sigma", self.posterior_scale(self.bias_rho).detach().clone())
        else:
            self.register_buffer("bias_prior_mu", None)
            self.register_buffer("bias_prior_sigma", None)
        self.register_buffer("uses_previous_posterior_prior", torch.tensor(False))

    @torch.no_grad()
    def snapshot_posterior_as_prior(self):
        """Freeze q_t as p_(t+1); buffers deliberately do not share storage."""
        self.weight_prior_mu.copy_(self.weight_mu)
        self.weight_prior_sigma.copy_(self.posterior_scale(self.weight_rho))
        if self.bias_mu is not None:
            self.bias_prior_mu.copy_(self.bias_mu)
            self.bias_prior_sigma.copy_(self.posterior_scale(self.bias_rho))
        self.uses_previous_posterior_prior.fill_(True)

    @staticmethod
    def _log_variational_posterior(value, mu, rho):
        sigma = _BayesianParameterLayer.posterior_scale(rho)
        return torch.distributions.Normal(mu, sigma).log_prob(value).sum()

    @staticmethod
    def posterior_scale(rho):
        """Finite positive posterior standard deviation used by all UCB layers."""
        rho = torch.nan_to_num(rho, nan=-3.0, posinf=UCB_MAX_RHO, neginf=UCB_MIN_RHO)
        return F.softplus(rho.clamp(UCB_MIN_RHO, UCB_MAX_RHO)).clamp_min(1e-6)

    def _prior_log_prob(self, value, prior_mu, prior_sigma):
        if self.uses_previous_posterior_prior:
            return torch.distributions.Normal(prior_mu, prior_sigma).log_prob(value).sum()
        return self.prior.log_prob(value)

    def _sample_parameter(self, mu, rho, prior_mu, prior_sigma, sample, calculate_log_probs):
        stochastic = self.training or sample
        value = mu + self.posterior_scale(rho) * torch.randn_like(mu) if stochastic else mu
        if self.training or calculate_log_probs:
            return value, self._prior_log_prob(value, prior_mu, prior_sigma), self._log_variational_posterior(value, mu, rho)
        return value, value.new_zeros(()), value.new_zeros(())


class BayesianConv1d(_BayesianParameterLayer):
    """Bayesian 1-D convolution equivalent to UCB/src's BayesianConv2D."""
    def __init__(self, in_channels, out_channels, kernel_size, *, dilation=1, bias=True,
                 rho=-3.0, sig1=0.0, sig2=6.0, pi=0.25):
        super().__init__(rho, sig1, sig2, pi)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size).normal_(0.0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty_like(self.weight_mu).normal_(rho, 0.1))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_channels).normal_(0.0, 0.1))
            self.bias_rho = nn.Parameter(torch.empty_like(self.bias_mu).normal_(rho, 0.1))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)
        self._initialize_continual_prior()

    def forward(self, x, sample=False, calculate_log_probs=False):
        weight, log_prior, log_q = self._sample_parameter(
            self.weight_mu, self.weight_rho, self.weight_prior_mu, self.weight_prior_sigma, sample, calculate_log_probs
        )
        if self.bias_mu is None:
            bias = None
        else:
            bias, bias_prior, bias_log_q = self._sample_parameter(
                self.bias_mu, self.bias_rho, self.bias_prior_mu, self.bias_prior_sigma, sample, calculate_log_probs
            )
            log_prior, log_q = log_prior + bias_prior, log_q + bias_log_q
        self.log_prior, self.log_variational_posterior = log_prior, log_q
        return F.conv1d(x, weight, bias, dilation=self.dilation)


class BayesianLinear(_BayesianParameterLayer):
    """Bayesian affine layer with the UCB/src variational parameterization."""
    def __init__(self, in_features, out_features, *, bias=True, rho=-3.0, sig1=0.0, sig2=6.0, pi=0.25):
        super().__init__(rho, sig1, sig2, pi)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0.0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty_like(self.weight_mu).normal_(rho, 0.1))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0.0, 0.1))
            self.bias_rho = nn.Parameter(torch.empty_like(self.bias_mu).normal_(rho, 0.1))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)
        self._initialize_continual_prior()

    def forward(self, x, sample=False, calculate_log_probs=False):
        weight, log_prior, log_q = self._sample_parameter(
            self.weight_mu, self.weight_rho, self.weight_prior_mu, self.weight_prior_sigma, sample, calculate_log_probs
        )
        if self.bias_mu is None:
            bias = None
        else:
            bias, bias_prior, bias_log_q = self._sample_parameter(
                self.bias_mu, self.bias_rho, self.bias_prior_mu, self.bias_prior_sigma, sample, calculate_log_probs
            )
            log_prior, log_q = log_prior + bias_prior, log_q + bias_log_q
        self.log_prior, self.log_variational_posterior = log_prior, log_q
        return F.linear(x, weight, bias)


class _BayesianCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **ucb_kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = BayesianConv1d(in_channels, out_channels, kernel_size, dilation=dilation, **ucb_kwargs)

    def forward(self, x, sample=False, calculate_log_probs=False):
        return self.conv(F.pad(x, (self.pad, 0)), sample, calculate_log_probs)


class _BayesianTCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout, **ucb_kwargs):
        super().__init__()
        self.conv1 = _BayesianCausalConv1d(channels, channels, kernel_size, dilation, **ucb_kwargs)
        self.conv2 = _BayesianCausalConv1d(channels, channels, kernel_size, dilation, **ucb_kwargs)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, sample=False, calculate_log_probs=False):
        out = self.dropout1(F.silu(self.conv1(x, sample, calculate_log_probs)))
        out = self.dropout2(F.silu(self.conv2(out, sample, calculate_log_probs)))
        return F.silu(out + x)


class UCBTCN(nn.Module):
    """Bayes-by-Backprop TCN using the variational UCB parameterization.

    ``sample=True`` draws one weight sample for Monte-Carlo prediction;
    ``calculate_log_probs=True`` records the prior and variational log density
    needed by an ELBO training loop.  Its returned tensors match ``TCN``.
    """
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64,
                 tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=0.2,
                 ucb_rho=-3.0, ucb_sig1=0.0, ucb_sig2=6.0, ucb_pi=0.25):
        super().__init__()
        self.num_features, self.window_size = num_features, window_size
        ucb_kwargs = dict(rho=ucb_rho, sig1=ucb_sig1, sig2=ucb_sig2, pi=ucb_pi)
        self.input_proj = _BayesianCausalConv1d(num_features, tcn_num_channels, 1, 1, **ucb_kwargs)
        self.tcn_backbone = nn.ModuleList([
            _BayesianTCNResidualBlock(tcn_num_channels, tcn_kernel_size, 2 ** index, tcn_dropout, **ucb_kwargs)
            for index in range(tcn_num_blocks)
        ])
        self.velocity_head = BayesianLinear(tcn_num_channels, 2 * len(VELOCITY_COMPONENTS), **ucb_kwargs)
        self.contact_head = BayesianLinear(tcn_num_channels, len(LEGS), **ucb_kwargs)

    def bayesian_log_probs(self):
        """Return summed log p(w) and log q(w), after a forward pass."""
        layers = [module for module in self.modules() if isinstance(module, _BayesianParameterLayer)]
        if any(layer.log_prior is None for layer in layers):
            raise RuntimeError("Call forward(..., calculate_log_probs=True) before requesting UCB log probabilities.")
        return sum(layer.log_prior for layer in layers), sum(layer.log_variational_posterior for layer in layers)

    @torch.no_grad()
    def snapshot_posterior_as_prior(self):
        """Turn the completed task posterior into the frozen prior for the next task."""
        for module in self.modules():
            if isinstance(module, _BayesianParameterLayer):
                module.snapshot_posterior_as_prior()

    def extract_features(self, x, sample=False, calculate_log_probs=False):
        """Return causal TCN features for latent-space diagnostics."""
        features = self.input_proj(x.permute(0, 2, 1), sample, calculate_log_probs)
        for block in self.tcn_backbone:
            features = block(features, sample, calculate_log_probs)
        return features

    def forward(self, x, return_sequence=True, sample=False, calculate_log_probs=False, **_):
        features = self.extract_features(x, sample, calculate_log_probs)
        if return_sequence:
            output = self.velocity_head(features.permute(0, 2, 1), sample, calculate_log_probs)
            velocity_seq = output[..., :3].permute(0, 2, 1).unsqueeze(1)
            covariance_seq = (F.softplus(output[..., 3:]) + MIN_VELOCITY_VARIANCE).permute(0, 2, 1).unsqueeze(1)
            velocity_out, covariance_out = velocity_seq[..., -1], covariance_seq[..., -1]
        else:
            output = self.velocity_head(features[:, :, -1], sample, calculate_log_probs)
            velocity_out = output[..., :3].unsqueeze(1)
            covariance_out = (F.softplus(output[..., 3:]) + MIN_VELOCITY_VARIANCE).unsqueeze(1)
            velocity_seq = covariance_seq = None
        contact_out = self.contact_head(features[:, :, -1], sample, calculate_log_probs)
        return velocity_seq, velocity_out, covariance_seq, covariance_out, contact_out


class CausalConv1d(nn.Module):
    """Causal convolution with weight normalization."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))  # pad only left
        return self.conv(x)

class contact_cnn(nn.Module):
    def __init__(self, window_size=10, num_features=12):
        super(contact_cnn, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            CausalConv1d(
                in_ch=num_features,
                out_ch=128,
                kernel_size=3,
                dilation=1
            ),
            nn.SiLU(),
        )
        
        self.conv2 = nn.Sequential(
            CausalConv1d(in_ch=128, out_ch=128, kernel_size=3, dilation=1),
            nn.SiLU(),
        )
        
        self.conv3 = nn.Sequential(
            CausalConv1d(
                in_ch=128,
                out_ch=128,
                kernel_size=3,
                dilation=1
            ),
            nn.SiLU(),
        )

        self.conv4 = nn.Sequential(
            CausalConv1d(128, 128, kernel_size=3, dilation=1),
            nn.SiLU(),
        )

        self.conv5 = nn.Sequential(
            CausalConv1d(128, 128, kernel_size=3, dilation=1),
            nn.SiLU(),
        )
        
        # Velocity prediction heads
        self.velocity_heads = make_velocity_heads(128)
        
        # Left-foot contact logit.
        self.contact_heads = make_contact_heads(128)

    def forward(self, x, return_sequence=True):
        # x shape: (batch_size, window_size, num_features) - RAW features from csv2numpy.py
        
        # Permute to (batch_size, num_features, window_size) for Conv1d
        x = x.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        
        # Convolutional layers
        x = self.conv1(x)  # [B, 64, T]
        x = self.conv2(x)  # [B, 64, T]
        x = self.conv3(x)  # [B, 64, T]
        x = self.conv4(x)  # [B, 64, T]
        x = self.conv5(x)  # [B, 64, T]
        features = x  # [B, 64, T]
        
        # Velocity prediction (sequence-to-sequence)
        velocity_seq, velocity_out, covariance_seq, covariance_out = predict_velocity(
            self.velocity_heads, features, return_sequence
        )
        
        # Contact prediction (MLP on last timestep features)
        features_last = features[:, :, -1]  # [B, 64]
        contact_out = torch.cat([self.contact_heads[leg](features_last) for leg in LEGS], dim=1)
        
        # Return both sequence (for training) and last timestep (for inference)
        # During training: use velocity_seq for dense supervision
        # During inference: use velocity_out (last timestep only)
        return velocity_seq, velocity_out, covariance_seq, covariance_out, contact_out


class ContactCNNWithNormalization(nn.Module):
    """
    Wrapper class that embeds global normalization statistics into the model.
    This wrapper can be exported to ONNX so normalization is done inside the model
    during inference, eliminating the need to normalize data externally in C++.
    
    Normalization strategy: Global z-score normalization
    - Uses mean and std computed from training data (stored as buffers)
    - Normalize: (x - global_mean) / (global_std + eps)
    - Statistics are saved with the model and exported to ONNX
    
    Input shape: (batch_size, window_size, num_features) - RAW left-leg features from csv2numpy.py
    Output shapes:
        - velocity_seq: (batch_size, 1, 3, window_size) - body-frame body velocities
        - velocity_out: (batch_size, 1, 3) - body-frame last-timestep velocity
        - covariance_seq: (batch_size, 1, 3, window_size) - diagonal body-velocity variances
        - covariance_out: (batch_size, 1, 3) - diagonal last-timestep body-velocity variances
        - contact_out: (batch_size, 1) - left-foot contact logit at last timestep
    
    Feature layout:
    - Input: RAW features from csv2numpy.py
    - Z-score normalization is applied to all input features
    """
    def __init__(self, base_model, global_mean=None, global_std=None, eps=1e-8):
        """
        Args:
            base_model: The contact_cnn model to wrap
            global_mean: Tensor of shape (1, 1, num_features) with training data mean per feature
            global_std: Tensor of shape (1, 1, num_features) with training data std per feature
            eps: Small constant to prevent division by zero
        """
        super(ContactCNNWithNormalization, self).__init__()
        self.base_model = base_model
        self.eps = eps
        
        # Get num_features from base model
        num_features = base_model.num_features
        
        # Register as buffers (not parameters - won't be trained, but saved with model and exported to ONNX)
        if global_mean is not None:
            self.register_buffer('global_mean', global_mean)
        else:
            # Fallback: no normalization if stats not provided
            self.register_buffer('global_mean', torch.zeros(1, 1, num_features))
            
        if global_std is not None:
            self.register_buffer('global_std', global_std)
        else:
            # Fallback: no normalization if stats not provided
            self.register_buffer('global_std', torch.ones(1, 1, num_features))
        
    def forward(self, x, return_sequence=False, return_posteriors=False, return_flow_nll=False,
                sample=False, calculate_log_probs=False):
        """
        Apply z-score normalization, then pass through base model.
        
        Args:
            x: Raw input data (batch_size, window_size, num_features) - NOT z-score normalized
        
        Returns:
            velocity_seq: (batch_size, 1, 3, window_size) - body-velocity predictions
            velocity_out: (batch_size, 1, 3) - last-timestep body-velocity prediction
            covariance_seq: (batch_size, 1, 3, window_size) - diagonal body-velocity variances
            covariance_out: (batch_size, 1, 3) - diagonal last-timestep body-velocity variances
            contact_out: (batch_size, 1) - left-foot contact logit at last timestep
        """
        # Apply global z-score normalization to all input features
        # These statistics are embedded in the model and exported to ONNX
        x_normalized = (x - self.global_mean) / (self.global_std + self.eps)
        
        # Pass normalized data through the base model
        if return_flow_nll:
            return self.base_model(
                x_normalized, return_sequence=return_sequence, return_flow_nll=True
            )
        if return_posteriors:
            return self.base_model(
                x_normalized, return_sequence=return_sequence, return_posteriors=True
            )
        if sample or calculate_log_probs:
            return self.base_model(
                x_normalized, return_sequence=return_sequence, sample=sample,
                calculate_log_probs=calculate_log_probs,
            )
        return self.base_model(x_normalized, return_sequence=return_sequence)
