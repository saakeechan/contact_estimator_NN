import torch
import torch.nn as nn


class ContactCNNWithNormalization(nn.Module):
    """Embed training z-score statistics around any temporal velocity model."""
    def __init__(self, base_model, global_mean=None, global_std=None, eps=1e-8):
        super().__init__()
        self.base_model, self.eps = base_model, eps
        features = base_model.num_features
        self.register_buffer('global_mean', torch.zeros(1, 1, features) if global_mean is None else global_mean.reshape(1, 1, features))
        self.register_buffer('global_std', torch.ones(1, 1, features) if global_std is None else global_std.reshape(1, 1, features))
    def forward(self, x, return_sequence=False, **kwargs):
        return self.base_model((x - self.global_mean) / (self.global_std + self.eps), return_sequence=return_sequence, **kwargs)
