import unittest

import torch
from torch import nn

from src.natpn import NatPNVelocityHeads
from natpn.distributions import NormalGammaPrior


class SelectSecondFeature(nn.Module):
    def forward(self, inputs):
        return inputs[:, 1]


class NatPNPriorWeightedMeanTest(unittest.TestCase):
    def test_evidence_weights_likelihood_and_prior_means(self):
        heads = NatPNVelocityHeads(channels=2, legs=('left',), flow_layers=1)
        heads.flow = SelectSecondFeature()
        heads.scaler = nn.Identity()
        with torch.no_grad():
            for outputs in heads.outputs.values():
                for output in outputs:
                    output.prior = NormalGammaPrior(mean=0., lambd=1. / 3., alpha=2., beta=1.5)
                    output.linear.weight.zero_()
                    output.linear.bias.zero_()
                    output.linear.weight[0, 0] = 1.0

        features = torch.tensor([[[3.0], [0.0]], [[3.0], [torch.log(torch.tensor(4.0))]]])
        _, means, _, _, posteriors = heads(features, return_sequence=False)

        self.assertTrue(torch.allclose(means[..., 0], torch.tensor([[2.25], [36.0 / 13.0]])))
        self.assertTrue(torch.allclose(posteriors[0][0].lambd, torch.tensor([4.0 / 3.0, 13.0 / 3.0])))
        self.assertTrue(torch.allclose(posteriors[0][0].beta, torch.tensor([3.125, 127.0 / 26.0])))


if __name__ == '__main__':
    unittest.main()
