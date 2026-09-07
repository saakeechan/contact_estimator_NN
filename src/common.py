import torch
import torch.nn as nn
import torch.nn.functional as F

VELOCITY_COMPONENTS = ("x", "y", "z")
CANONICAL_LEGS = ("left", "right")
MIN_VELOCITY_VARIANCE = 1e-6


def resolve_active_legs(value):
    value = str(value).lower()
    if value == 'both': return CANONICAL_LEGS
    if value in CANONICAL_LEGS: return (value,)
    raise ValueError("active_legs must be 'left', 'right', or 'both'.")


def make_contact_heads(channels, legs):
    return nn.ModuleDict({leg: nn.Sequential(nn.Linear(channels, 256), nn.SiLU(), nn.Linear(256, 32), nn.SiLU(), nn.Linear(32, 1)) for leg in legs})


def contact_logits(heads, features, legs):
    return torch.cat([heads[leg](features[:, :, -1]) for leg in legs], dim=1)


class TemporalModel(nn.Module):
    """Shared model contract; subclasses provide an encoder and velocity head."""
    def __init__(self, window_size, num_features, legs):
        super().__init__()
        self.num_features, self.window_size = num_features, window_size
        self.legs = tuple(legs)
