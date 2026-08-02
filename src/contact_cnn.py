import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

import numpy as np

VELOCITY_COMPONENTS = ("x", "y", "z")
LEGS = ("left",)
MIN_VELOCITY_VARIANCE = 1e-6

NATPN_ROOT = Path(__file__).resolve().parents[1] / "NATPN" / "natural-posterior-network"
if not NATPN_ROOT.is_dir():
    raise RuntimeError(f"Bundled NatPN source is missing: {NATPN_ROOT}")
if str(NATPN_ROOT) not in sys.path:
    sys.path.insert(0, str(NATPN_ROOT))

from natpn.nn import NaturalPosteriorNetworkModel
from natpn.nn.flow import RadialFlow
from natpn.nn.output import NormalOutput
from natpn.nn.scaler import EvidenceScaler
from natpn.distributions.normal import NormalGamma


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
    """Task NatPN or decoupled task likelihood plus input-latent epistemic evidence."""
    def __init__(self, latent_dim, evidence_source="task", flow_layers=8, certainty_budget="normal", input_epistemic_scale=1.0):
        super().__init__()
        self.evidence_source = evidence_source
        self.input_epistemic_scale = input_epistemic_scale
        if evidence_source == "input" and input_epistemic_scale <= 0:
            raise ValueError('input_epistemic_scale must be positive.')
        if evidence_source == "task":
            self.models = nn.ModuleList([
                NaturalPosteriorNetworkModel(
                    latent_dim=latent_dim, encoder=nn.Identity(),
                    flow=RadialFlow(latent_dim, flow_layers), output=NormalOutput(latent_dim),
                    certainty_budget=certainty_budget,
                ) for _ in VELOCITY_COMPONENTS
            ])
        else:
            self.outputs = nn.ModuleList([NormalOutput(latent_dim) for _ in VELOCITY_COMPONENTS])

    @staticmethod
    def predictive_variance(posterior):
        aleatoric = posterior.beta / (posterior.alpha - 1.0).clamp_min(MIN_VELOCITY_VARIANCE)
        return aleatoric + aleatoric / posterior.lambd

    def flow_nll(self, features, return_sequence):
        """Fit the original task-latent flows only in task-evidence mode."""
        if self.evidence_source != "task":
            raise RuntimeError('The input-latent flow is pretrained by trainEncoder.py.')
        inputs = features.permute(0, 2, 1).reshape(-1, features.shape[1]) if return_sequence else features[:, :, -1]
        return -torch.stack([
            model.log_prob(inputs.detach(), track_encoder_gradients=False).mean() for model in self.models
        ]).mean()

    def forward(self, features, log_evidence, return_sequence):
        if return_sequence:
            batch_size, _, num_steps = features.shape
            inputs = features.permute(0, 2, 1).reshape(-1, features.shape[1])
            log_evidence = log_evidence.reshape(-1)
        else:
            batch_size, _, num_steps = features.shape
            inputs = features[:, :, -1]

        if self.evidence_source == "task":
            posteriors = [model(inputs)[0] for model in self.models]
            means = torch.stack([posterior.maximum_a_posteriori().mean() for posterior in posteriors], dim=-1)
        else:
            likelihoods = [output(inputs) for output in self.outputs]
            evidence = log_evidence.exp()
            aleatoric_variances = [likelihood.precision.reciprocal() for likelihood in likelihoods]
            epistemic_variances = [
                torch.full_like(evidence, self.input_epistemic_scale) / (output.prior.evidence + evidence)
                for output in self.outputs
            ]
            posteriors = [
                NormalGamma(
                    likelihood.mean(),
                    output.prior.evidence + evidence,
                    torch.ones_like(evidence) * output.prior.alpha,
                    aleatoric * (output.prior.alpha - 1.0),
                )
                for output, likelihood, aleatoric in zip(self.outputs, likelihoods, aleatoric_variances)
            ]
            for posterior, epistemic in zip(posteriors, epistemic_variances):
                posterior.epistemic_variance = epistemic
            means = torch.stack([likelihood.mean() for likelihood in likelihoods], dim=-1)
            variances = torch.stack([aleatoric + epistemic for aleatoric, epistemic in zip(aleatoric_variances, epistemic_variances)], dim=-1)
        if self.evidence_source == "task":
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
                nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 4 * len(VELOCITY_COMPONENTS))
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
                 natpn_flow_layers=8, natpn_certainty_budget="normal", natpn_evidence_source="task",
                 input_natpn_checkpoint=None, input_epistemic_scale=1.0):
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
        
        if natpn_evidence_source not in {"task", "input"}:
            raise ValueError("natpn_evidence_source must be 'task' or 'input'.")
        self.natpn_evidence_source = natpn_evidence_source
        if natpn_evidence_source == "input":
            if not input_natpn_checkpoint:
                raise ValueError('input_natpn_checkpoint is required when natpn_evidence_source is input.')
            input_natpn_checkpoint = (
                input_natpn_checkpoint if os.path.isabs(input_natpn_checkpoint)
                else str(Path(__file__).resolve().parents[1] / input_natpn_checkpoint)
            )
            self.input_density = FrozenInputLatentDensity(input_natpn_checkpoint)
        self.velocity_heads = NatPNVelocityHeads(
            tcn_num_channels, natpn_evidence_source, natpn_flow_layers, natpn_certainty_budget, input_epistemic_scale
        )
        
        # 4. Left-foot contact logit.
        self.contact_heads = make_contact_heads(tcn_num_channels)
    
    def forward(self, x, return_sequence=True, return_posteriors=False, return_flow_nll=False,
                input_density_x=None):
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
        if self.natpn_evidence_source == "input" and input_density_x is None:
            raise ValueError('input_density_x must be the raw window used by the frozen input encoder.')
        if self.natpn_evidence_source == "input" and return_sequence:
            raise ValueError('Input-latent NatPN evidence is trained only for final-timestep supervision.')
        
        # 1. Convert to Conv1d format
        x = x.permute(0, 2, 1)  # [B, T, F] → [B, F, T]
        
        # 2. Project to TCN channel dimension
        z = self.input_proj(x)  # [B, F, T] → [B, tcn_num_channels, T]
        
        # 3. TCN backbone
        features = self.tcn_backbone(z)  # [B, tcn_num_channels, T]
        if return_flow_nll:
            return self.input_density.nll(input_density_x) if self.natpn_evidence_source == "input" else self.velocity_heads.flow_nll(features, return_sequence)
        log_evidence = self.input_density.log_evidence(input_density_x) if self.natpn_evidence_source == "input" else None
        
        # 4. Velocity prediction
        velocity_seq, velocity_out, covariance_seq, covariance_out, posteriors = self.velocity_heads(
            features, log_evidence, return_sequence
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


class DenoisingTCNAutoencoder(nn.Module):
    """Reconstruct raw input windows from noisy versions using the TCN backbone."""
    def __init__(self, window_size, num_features, tcn_num_channels=64,
                 tcn_kernel_size=3, tcn_num_blocks=3, tcn_dropout=0.2,
                 global_mean=None, global_std=None, eps=1e-8):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.eps = eps
        self.register_buffer(
            'global_mean', torch.zeros(1, 1, num_features) if global_mean is None else global_mean
        )
        self.register_buffer(
            'global_std', torch.ones(1, 1, num_features) if global_std is None else global_std
        )
        self.input_proj = nn.Conv1d(num_features, tcn_num_channels, kernel_size=1)
        self.tcn_backbone = nn.Sequential(*[
            TCNResidualBlock(tcn_num_channels, tcn_num_channels, tcn_kernel_size, 2 ** i, tcn_dropout)
            for i in range(tcn_num_blocks)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(tcn_num_channels, tcn_num_channels),
            nn.SiLU(),
            nn.Linear(tcn_num_channels, tcn_num_channels),
            nn.SiLU(),
            nn.Linear(tcn_num_channels, num_features)
        )

    def forward(self, noisy_x):
        x = (noisy_x - self.global_mean) / (self.global_std + self.eps)
        features = self.tcn_backbone(self.input_proj(x.permute(0, 2, 1)))
        reconstruction_normalized = self.decoder(features.permute(0, 2, 1))
        return reconstruction_normalized * (self.global_std + self.eps) + self.global_mean


class FrozenInputLatentDensity(nn.Module):
    """Load the reconstruction encoder and NatPN flow produced by trainEncoder.py."""
    def __init__(self, checkpoint_path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder_type = checkpoint['encoder_type']
        encoder_config = checkpoint.get('encoder_config', {})
        self.encoder = DenoisingTCNAutoencoder(
            window_size=encoder_config.get('window_size', 1),
            num_features=checkpoint['num_features'],
            tcn_num_channels=checkpoint['latent_dim'],
            tcn_kernel_size=encoder_config.get('tcn_kernel_size', 3),
            tcn_num_blocks=encoder_config.get('tcn_num_blocks', len({
                key.split('.')[1] for key in checkpoint['encoder_state_dict']
                if key.startswith('tcn_backbone.') and key.endswith('conv1.conv.bias')
            })),
            tcn_dropout=encoder_config.get('tcn_dropout', 0.2),
        )
        self.is_vae = encoder_type == 'VAE'
        if self.is_vae:
            self.mu = nn.Conv1d(checkpoint['latent_dim'], checkpoint['latent_dim'], kernel_size=1)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=not self.is_vae)
        if self.is_vae:
            self.mu.load_state_dict({
                key.removeprefix('mu.'): value
                for key, value in checkpoint['encoder_state_dict'].items() if key.startswith('mu.')
            })
        self.flow = RadialFlow(checkpoint['latent_dim'], checkpoint['input_natpn_flow_layers'])
        self.scaler = EvidenceScaler(checkpoint['latent_dim'], checkpoint['input_natpn_certainty_budget'])
        self.flow.load_state_dict({
            key.removeprefix('flow.'): value
            for key, value in checkpoint['input_natpn_state_dict'].items() if key.startswith('flow.')
        })
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _latent(self, raw_window):
        normalized = (raw_window - self.encoder.global_mean) / (self.encoder.global_std + self.encoder.eps)
        features = self.encoder.tcn_backbone(self.encoder.input_proj(normalized.permute(0, 2, 1)))
        return self.mu(features) if self.is_vae else features

    def log_evidence(self, raw_window):
        self.eval()
        with torch.no_grad():
            return self.scaler(self.flow(self._latent(raw_window)[:, :, -1]))

    def nll(self, raw_window):
        self.eval()
        with torch.no_grad():
            return -self.flow(self._latent(raw_window)[:, :, -1]).mean()


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
        
    def forward(self, x, return_sequence=False, return_posteriors=False, return_flow_nll=False):
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
                x_normalized, return_sequence=return_sequence, return_flow_nll=True, input_density_x=x
            )
        if return_posteriors:
            return self.base_model(
                x_normalized, return_sequence=return_sequence, return_posteriors=True, input_density_x=x
            )
        return self.base_model(x_normalized, return_sequence=return_sequence, input_density_x=x)
