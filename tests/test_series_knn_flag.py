import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.base_test_series import BaseSeriesTest


class NoKnnSeries(BaseSeriesTest):
    single_test_script = 'src/testSingleNatPN.py'
    run_knn = False


class InProcessSeries(BaseSeriesTest):
    single_test_script = 'unused.py'

    def create_in_process_runner(self, args, root, knn_cache_path):
        return lambda seed: {
            'seed': seed, 'ood_feature': 'cmd_vel', 'environment': None, 'cmd_vel_x': 1.0,
            'velocity_mae': 0.1, 'uncertainty_final_timestep_gt_contact': True,
            'aleatoric_variance': [1.0, 1.0, 1.0], 'epistemic_variance': [1.0, 1.0, 1.0],
            'negative_log_density': 1.0, 'knn_ood_windows': None,
            'knn_total_windows': None, 'knn_ood_percentage_total_windows': None,
        }


class SeriesKnnFlagTest(unittest.TestCase):
    def test_disabled_knn_is_forwarded_and_written_as_not_run(self):
        commands = []

        def run(command, **_):
            commands.append(command)
            if '--metrics-json' in command:
                path = Path(command[command.index('--metrics-json') + 1])
                path.write_text(json.dumps({
                    'seed': 1, 'ood_feature': 'cmd_vel', 'environment': None, 'cmd_vel_x': 1.0,
                    'velocity_mae': 0.1, 'uncertainty_final_timestep_gt_contact': True,
                    'aleatoric_variance': [1.0, 1.0, 1.0], 'epistemic_variance': [1.0, 1.0, 1.0],
                    'negative_log_density': 1.0, 'knn_ood_windows': None,
                    'knn_total_windows': None, 'knn_ood_percentage_total_windows': None,
                }))

        with tempfile.TemporaryDirectory() as directory:
            output_csv = Path(directory) / 'metrics.csv'
            with patch('utils.base_test_series.subprocess.run', side_effect=run):
                NoKnnSeries().main(['--first-seed', '1', '--last-seed', '1', '--output-csv', str(output_csv)])
            with output_csv.open(newline='') as file:
                row = next(csv.DictReader(file))

        self.assertIn('--skip-knn', commands[0])
        self.assertNotIn('--knn-cache', commands[0])
        self.assertEqual(row['knn_ood_percentage_total_windows'], '')

    def test_in_process_runner_replaces_per_seed_subprocesses(self):
        with tempfile.TemporaryDirectory() as directory:
            output_csv = Path(directory) / 'metrics.csv'
            with patch('utils.base_test_series.subprocess.run') as run:
                InProcessSeries().main(['--first-seed', '1', '--last-seed', '2', '--output-csv', str(output_csv)])

        self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
