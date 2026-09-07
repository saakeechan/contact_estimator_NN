import torch.nn as nn
try:
    from .common import contact_logits, make_contact_heads, TemporalModel
    from .deterministic import CausalConv1d
    from .ensemble import make_velocity_heads, predict_velocity
except ImportError:
    from common import contact_logits, make_contact_heads, TemporalModel
    from deterministic import CausalConv1d
    from ensemble import make_velocity_heads, predict_velocity


class contact_cnn(TemporalModel):
    def __init__(self, window_size=10, num_features=12, legs=('left', 'right')):
        super().__init__(window_size, num_features, legs)
        self.conv1 = nn.Sequential(CausalConv1d(num_features, 128), nn.SiLU()); self.conv2 = nn.Sequential(CausalConv1d(128, 128), nn.SiLU()); self.conv3 = nn.Sequential(CausalConv1d(128, 128), nn.SiLU()); self.conv4 = nn.Sequential(CausalConv1d(128, 128), nn.SiLU()); self.conv5 = nn.Sequential(CausalConv1d(128, 128), nn.SiLU())
        self.velocity_heads, self.contact_heads = make_velocity_heads(128, self.legs), make_contact_heads(128, self.legs)
    def forward(self, x, return_sequence=True):
        features = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x.permute(0, 2, 1))))))
        return (*predict_velocity(self.velocity_heads, features, self.legs, return_sequence), contact_logits(self.contact_heads, features, self.legs))
