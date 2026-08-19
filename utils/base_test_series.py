"""Shared seed-series runner for model-specific single-test scripts."""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


class BaseSeriesTest:
    """Run a concrete single-test script over a seed range and collect metrics."""

    description = 'Run a model-specific single-test evaluator over a seed range and collect its metrics in a CSV.'
    single_test_script = None
    output_prefix = 'tests'
    default_ood_feature = None
    default_cmd_vel_x_window = (1.5, 3.0)
    default_environment_windows = None
    default_plot_metric = 'epistemic'
    run_knn = True
    first_seed = 101
    last_seed = 200

    def main(self, argv=None):
        root = Path(__file__).resolve().parents[1]
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('--config-name', default=root / 'config/network_params.yaml', type=Path)
        parser.add_argument('--cmd-vel-x-window', type=float, nargs=2, metavar=('MIN', 'MAX'),
                            default=self.default_cmd_vel_x_window)
        parser.add_argument('--ood-feature', choices=('cmd_vel', 'environment'), default=self.default_ood_feature,
                            help='Override ood_feature from the config.')
        parser.add_argument('--environment-window', type=int, nargs=2, action='append', metavar=('MIN', 'MAX'),
                            default=self.default_environment_windows,
                            help='Inclusive environment window; repeat this option for disjoint windows.')
        parser.add_argument('--first-seed', type=int, default=self.first_seed)
        parser.add_argument('--last-seed', type=int, default=self.last_seed)
        parser.add_argument('--output-csv', type=Path)
        parser.add_argument('--with-umap', action='store_true', help='Also save one UMAP figure per seed.')
        parser.add_argument('--skip-knn', action='store_true', default=not self.run_knn,
                            help='Do not run kNN/OOD evaluation for any seed.')
        parser.add_argument('--plot-metric', choices=('epistemic', 'negative-log-density'),
                            default=self.default_plot_metric, help='Y-axis metric for the summary plot.')
        args = parser.parse_args(argv)
        if args.first_seed > args.last_seed:
            parser.error('--first-seed must be no greater than --last-seed')
        if args.output_csv is None:
            args.output_csv = root / 'testResults' / self.output_csv_name(args)

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        with tempfile.TemporaryDirectory() as temp_dir:
            knn_cache_path = Path(temp_dir) / 'knn_reference.npz'
            in_process_runner = self.create_in_process_runner(args, root, knn_cache_path)
            for seed in range(args.first_seed, args.last_seed + 1):
                if in_process_runner is not None:
                    print(f'Running seed {seed}')
                    rows.append(self.row_from_metrics(in_process_runner(seed)))
                    continue
                metrics_path = Path(temp_dir) / f'{seed}.json'
                command = [
                    sys.executable, root / self.single_test_script,
                    '--config_name', args.config_name, '--seed', str(seed),
                    '--cmd-vel-x-window', *(str(value) for value in args.cmd_vel_x_window),
                    '--metrics-json', metrics_path, '--skip-plots',
                    '--save-umap' if args.with_umap else '--skip-umap',
                ]
                if args.skip_knn:
                    command.append('--skip-knn')
                else:
                    command.extend(('--knn-cache', knn_cache_path))
                if args.ood_feature:
                    command.extend(('--ood-feature', args.ood_feature))
                for low, high in args.environment_window or ():
                    command.extend(('--environment-window', str(low), str(high)))
                print(f'Running seed {seed}')
                subprocess.run(command, cwd=root, check=True)
                with open(metrics_path) as metrics_file:
                    rows.append(self.row_from_metrics(json.load(metrics_file)))

        with open(args.output_csv, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows)} rows to {args.output_csv}')
        subprocess.run([
            sys.executable, root / 'utils/make_testsingle_table.py', '--input-csv', args.output_csv,
            '--plot-metric', args.plot_metric,
        ], cwd=root, check=True)

    @staticmethod
    def row_from_metrics(metrics):
        scientific = lambda value: '' if value is None else f'{float(value):.4e}'
        return {
            'seed': metrics['seed'], 'ood_feature': metrics['ood_feature'],
            'environment': metrics['environment'], 'cmd_vel_x': scientific(metrics['cmd_vel_x']),
            'velocity_mae': scientific(metrics['velocity_mae']),
            'uncertainty_final_timestep_gt_contact': int(metrics['uncertainty_final_timestep_gt_contact']),
            **{f'aleatoric_v{axis}': scientific(value) for axis, value in zip('xyz', metrics['aleatoric_variance'])},
            **{f'epistemic_v{axis}': scientific(value) for axis, value in zip('xyz', metrics['epistemic_variance'])},
            # Only NatPN exposes task-latent flow negative log density.
            'negative_log_density': scientific(metrics.get('negative_log_density')),
            'knn_ood_windows': metrics['knn_ood_windows'], 'knn_total_windows': metrics['knn_total_windows'],
            'knn_ood_percentage_total_windows': scientific(metrics['knn_ood_percentage_total_windows']),
        }

    def output_csv_name(self, args):
        return f'{self.output_prefix}_seeds_{args.first_seed}-{args.last_seed}.csv'

    def create_in_process_runner(self, args, root, knn_cache_path):
        """Return a seed-to-metrics callable, or None to retain subprocess evaluation."""
        return None
