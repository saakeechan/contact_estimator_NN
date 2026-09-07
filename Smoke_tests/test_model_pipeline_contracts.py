"""Fast CPU checks for model outputs, leg ordering, and trainer losses."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common import resolve_active_legs
from src.der import DERTCN
from src.ensemble import EnsembleTCN
from src.mc_dropout import MCDropoutTCN
from src.natpn import NormalOutput, TCN as NatPNTCN
from src.ucb import UCBTCN
from src.vanilla_cnn import contact_cnn
from src.vcl import VCLTCN
from train.trainDER import DERTrainer, nig_loss
from train.trainEnsemble import EnsembleTrainer
from train.trainMCdropout import MCDropoutTrainer
from train.trainNatPN import NatPNTrainer
from train.trainReplay import ReplayTrainer
from tests.testNatPN import NatPNSingleTest
from utils.data_handler import contact_dataset


BATCH, STEPS, FEATURES = 2, 5, 7


class ModelPipelineContractsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.inputs = torch.randn(BATCH, STEPS, FEATURES)
        self.target = torch.randn(BATCH, 2, 3)
        self.target_sequence = torch.randn(BATCH, STEPS, 2, 3)

    def assert_output_contract(self, model, legs, has_parameters=False):
        model.eval()
        kwargs = {'return_posteriors': True} if has_parameters else {}
        dense = model(self.inputs, return_sequence=True, **kwargs)
        final = model(self.inputs, return_sequence=False, **kwargs)
        expected_sequence = (BATCH, len(legs), 3, STEPS)
        expected_final = (BATCH, len(legs), 3)
        self.assertEqual(tuple(dense[0].shape), expected_sequence)
        self.assertEqual(tuple(dense[1].shape), expected_final)
        self.assertEqual(tuple(dense[2].shape), expected_sequence)
        self.assertEqual(tuple(dense[3].shape), expected_final)
        self.assertEqual(tuple(dense[4].shape), (BATCH, len(legs)))
        self.assertIsNone(final[0])
        self.assertEqual(tuple(final[1].shape), expected_final)
        self.assertIsNone(final[2])
        self.assertEqual(tuple(final[3].shape), expected_final)
        self.assertEqual(tuple(final[4].shape), (BATCH, len(legs)))
        self.assertTrue(torch.isfinite(dense[1]).all())
        self.assertTrue(torch.isfinite(dense[3]).all())
        self.assertTrue((dense[2] > 0).all())
        self.assertTrue(torch.allclose(dense[1], final[1], rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(dense[3], final[3], rtol=1e-5, atol=1e-6))
        if has_parameters:
            self.assertEqual(len(dense), 6)
            self.assertEqual(len(final[5]), len(legs))

    def test_all_models_share_the_output_contract_for_both_legs(self):
        constructors = (
            EnsembleTCN(STEPS, FEATURES, 4, 3, 1, 0., ('left', 'right')),
            MCDropoutTCN(STEPS, FEATURES, 4, 3, 1, .2, ('left', 'right')),
            DERTCN(STEPS, FEATURES, 4, 3, 1, 0., ('left', 'right')),
            UCBTCN(STEPS, FEATURES, 4, 3, 1, 0., legs=('left', 'right')),
            VCLTCN(STEPS, FEATURES, 4, 3, 1, 0., legs=('left', 'right')),
            contact_cnn(STEPS, FEATURES, ('left', 'right')),
        )
        for model in constructors:
            with self.subTest(model=type(model).__name__):
                self.assert_output_contract(model, ('left', 'right'))
        self.assert_output_contract(NatPNTCN(STEPS, FEATURES, 4, 3, 1, 0., 'radial', 1, 'constant', ('left', 'right')), ('left', 'right'), has_parameters=True)

    def test_single_leg_selection_controls_all_output_dimensions(self):
        for leg in ('left', 'right'):
            for model in (EnsembleTCN(STEPS, FEATURES, 4, 3, 1, 0., (leg,)), DERTCN(STEPS, FEATURES, 4, 3, 1, 0., (leg,)), UCBTCN(STEPS, FEATURES, 4, 3, 1, 0., legs=(leg,))):
                with self.subTest(model=type(model).__name__, leg=leg):
                    self.assertEqual(model.legs, (leg,))
                    self.assert_output_contract(model, (leg,))

    def test_wrong_input_feature_count_and_natpn_architecture_are_rejected(self):
        model = EnsembleTCN(STEPS, FEATURES, 4, 3, 1, 0., ('left', 'right'))
        with self.assertRaises(RuntimeError):
            model(torch.randn(BATCH, STEPS, FEATURES + 1))
        with self.assertRaisesRegex(ValueError, 'model_architecture'):
            NatPNSingleTest().build_model({'model_architecture': 'vanilla_cnn'}, FEATURES)

    def test_leg_head_order_is_left_then_right_and_matches_tensor_axes(self):
        for model, heads in (
            (EnsembleTCN(STEPS, FEATURES, 4, 3, 1, 0., ('left', 'right')), 'velocity_heads'),
            (DERTCN(STEPS, FEATURES, 4, 3, 1, 0., ('left', 'right')), 'velocity_heads.heads'),
        ):
            with self.subTest(model=type(model).__name__):
                velocity_heads = model.velocity_heads if heads == 'velocity_heads' else model.velocity_heads.heads
                self.assertEqual(tuple(velocity_heads.keys()), ('left', 'right'))
                self.assertEqual(tuple(model.contact_heads.keys()), ('left', 'right'))
                for index, leg in enumerate(model.legs):
                    final_layer = velocity_heads[leg][-1]
                    final_layer.weight.data.zero_()
                    final_layer.bias.data[:3].fill_(index + 1.)
                    model.contact_heads[leg][-1].weight.data.zero_()
                    model.contact_heads[leg][-1].bias.data.fill_(index + 1.)
                outputs = model(self.inputs, return_sequence=False,
                                **({'return_posteriors': True} if isinstance(model, DERTCN) else {}))
                self.assertTrue(torch.equal(outputs[1][0, :, 0], torch.tensor([1., 2.])))
                self.assertTrue(torch.equal(outputs[4][0], torch.tensor([1., 2.])))

    def test_ucb_shared_heads_preserve_leg_axis_order(self):
        model = UCBTCN(STEPS, FEATURES, 4, 3, 1, 0., legs=('left', 'right')).eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        model.velocity_head.bias_mu.data.copy_(torch.tensor([
            1., 1., 1., 0., 0., 0., 2., 2., 2., 0., 0., 0.,
        ]))
        model.contact_head.bias_mu.data.copy_(torch.tensor([3., 4.]))
        outputs = model(self.inputs, return_sequence=False)
        self.assertTrue(torch.equal(outputs[1][0, :, 0], torch.tensor([1., 2.])))
        self.assertTrue(torch.equal(outputs[4][0], torch.tensor([3., 4.])))

    def test_gaussian_and_der_trainers_use_their_declared_losses(self):
        gaussian_model = EnsembleTCN(STEPS, FEATURES, 4, 3, 1, 0., ('left', 'right')).eval()
        outputs = gaussian_model(self.inputs, return_sequence=False)
        expected = F.gaussian_nll_loss(outputs[1], self.target, outputs[3], reduction='none')
        expected.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in gaussian_model.parameters()))
        for trainer in (EnsembleTrainer, MCDropoutTrainer, ReplayTrainer):
            with self.subTest(trainer=trainer.__name__):
                self.assertTrue(torch.allclose(trainer.velocity_loss(object(), outputs, self.target, False), expected))

        der_model = DERTCN(STEPS, FEATURES, 4, 3, 1, 0., ('left', 'right')).eval()
        der_outputs = der_model(self.inputs, return_sequence=False, return_posteriors=True)
        gamma, nu, alpha, beta = der_outputs[5]
        expected_der = (.5 * torch.log(torch.pi / nu) - alpha * torch.log(2. * beta * (1. + nu))
                        + (alpha + .5) * torch.log(nu * (self.target - gamma).square() + 2. * beta * (1. + nu))
                        + torch.lgamma(alpha) - torch.lgamma(alpha + .5)
                        + .25 * (self.target - gamma).abs() * (2. * nu + alpha))
        self.assertTrue(torch.allclose(nig_loss((gamma, nu, alpha, beta), self.target, .25), expected_der))
        der_trainer = object.__new__(DERTrainer)
        der_trainer.regularizer_weight = .25
        self.assertTrue(torch.allclose(DERTrainer.velocity_loss(der_trainer, der_outputs, self.target, False), expected_der))

    def test_natpn_gaussian_loss_uses_final_or_dense_targets_without_axis_swaps(self):
        model = NatPNTCN(STEPS, FEATURES, 4, 3, 1, 0., 'radial', 1, 'constant', ('left', 'right')).eval()
        trainer = object.__new__(NatPNTrainer)
        trainer.loss_type = 'gaussian_nll'
        final_outputs = model(self.inputs, return_sequence=False, return_posteriors=True)
        expected_final = F.gaussian_nll_loss(final_outputs[1], self.target, final_outputs[3], reduction='none')
        self.assertTrue(torch.allclose(NatPNTrainer.velocity_loss(trainer, final_outputs, self.target, False), expected_final))
        dense_outputs = model(self.inputs, return_sequence=True, return_posteriors=True)
        expected_dense = F.gaussian_nll_loss(dense_outputs[0].permute(0, 3, 1, 2), self.target_sequence, dense_outputs[2].permute(0, 3, 1, 2), reduction='none')
        self.assertTrue(torch.allclose(NatPNTrainer.velocity_loss(trainer, dense_outputs, self.target_sequence, True), expected_dense))

    def test_dataset_shapes_and_leg_metadata_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.zeros((7, FEATURES), dtype=np.float32)
            np.save(root / 'all_data.npy', data)
            np.save(root / 'all_labels.npy', np.zeros((7, 1), dtype=np.int64))
            np.save(root / 'all_body_velocities.npy', np.zeros((7, 1, 3), dtype=np.float32))
            np.save(root / 'all_data_metadata.npy', {'legs': ['right'], 'num_features': FEATURES})
            dataset = contact_dataset(str(root / 'all_data.npy'), str(root / 'all_labels.npy'), 3, device='cpu')
            self.assertEqual(dataset.legs, ('right',))
            self.assertEqual(tuple(dataset[0]['velocity'].shape), (3, 1, 3))
            np.save(root / 'all_labels.npy', np.zeros((7, 2), dtype=np.int64))
            with self.assertRaisesRegex(ValueError, 'Expected'):
                contact_dataset(str(root / 'all_data.npy'), str(root / 'all_labels.npy'), 3, device='cpu')

    def test_leg_config_and_natpn_trainer_import_are_unambiguous(self):
        self.assertEqual(resolve_active_legs('both'), ('left', 'right'))
        self.assertEqual(resolve_active_legs('right'), ('right',))
        with self.assertRaises(ValueError):
            resolve_active_legs('front')
        self.assertIsNotNone(NatPNTrainer)

    def test_library_and_project_natpn_prior_defaults_are_intentionally_distinct(self):
        self.assertAlmostEqual(NormalOutput(2).prior.evidence.item(), 1. / 3.)
        model = NatPNTCN(STEPS, FEATURES, 4, 3, 1, 0., 'radial', 1, 'constant', ('left', 'right'))
        for outputs in model.velocity_heads.outputs.values():
            for output in outputs:
                self.assertAlmostEqual(output.prior.evidence.item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
