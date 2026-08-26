import unittest

from tests.base_testSeries import SeriesTest


class SeriesMetricsTest(unittest.TestCase):
    def test_skipped_knn_serializes_as_blank(self):
        row = SeriesTest.row_from_metrics({
            'seed': 1, 'ood_feature': 'cmd_vel', 'environment': None, 'cmd_vel_x': 1.0,
            'velocity_mae': 0.1, 'uncertainty_final_timestep_gt_contact': True,
            'aleatoric_variance': [1.0, 1.0, 1.0], 'epistemic_variance': [1.0, 1.0, 1.0],
            'knn_ood_windows': None, 'knn_total_windows': None, 'knn_ood_percentage_total_windows': None,
        })
        self.assertEqual(row['knn_ood_percentage_total_windows'], '')

    def test_model_choices_include_ensemble(self):
        parser = SeriesTest().build_parser(__import__('pathlib').Path.cwd())
        self.assertEqual(parser.parse_args(['--model', 'ensemble']).model, 'ensemble')


if __name__ == '__main__':
    unittest.main()
