import unittest

import torch
from torch import nn

from src.contact_cnn import NatPNVelocityHeads


class SelectSecondFeature(nn.Module):
    def forward(self, inputs):
        return inputs[:, 1]


class NatPNDirectMeanTest(unittest.TestCase):
    def test_evidence_changes_uncertainty_not_mean(self):
        heads = NatPNVelocityHeads(latent_dim=2, flow_layers=1)
        heads.flow = SelectSecondFeature()
        heads.scaler = nn.Identity()
        with torch.no_grad():
            for output in heads.outputs:
                output.linear.weight.zero_()
                output.linear.bias.zero_()
                output.linear.weight[0, 0] = 1.0

        features = torch.tensor([[[3.0], [0.0]], [[3.0], [torch.log(torch.tensor(4.0))]]])
        _, means, _, _, posteriors = heads(features, return_sequence=False)

        self.assertTrue(torch.allclose(means[..., 0], torch.tensor([[3.0], [3.0]])))
        self.assertTrue(torch.allclose(posteriors[0].lambd, torch.tensor([4.0 / 3.0, 13.0 / 3.0])))
        self.assertTrue(torch.allclose(posteriors[0].beta, torch.tensor([1.5, 3.0])))


if __name__ == '__main__':
    unittest.main()
