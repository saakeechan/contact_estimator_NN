"""Run one configurable NatPN, DER, or deep-ensemble evaluation seed sweep."""
import argparse
import csv
from argparse import Namespace
import json
import subprocess
import sys
import tempfile
from pathlib import Path


# Edit these defaults, or override them on the command line.
MODEL = 'natpn'  # 'natpn', 'der', or 'ensemble'
FIRST_SEED = 3301
LAST_SEED = 3500
OOD_FEATURE = 'cmd_vel'
CMD_VEL_X_WINDOW = (0.0, 3.0)
ENVIRONMENT_WINDOWS = [(0, 20)]
PLOT_METRIC = 'epistemic'
RUN_KNN = False

MODEL_SCRIPTS = {
    'natpn': 'tests/testNatPN.py',
    'der': 'tests/testDER.py',
    'ensemble': 'tests/testEnsemble.py',
}


class SeriesTest:
    def build_parser(self, root):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--config-name', default=root / 'config/network_params.yaml', type=Path)
        parser.add_argument('--model', choices=MODEL_SCRIPTS, default=MODEL)
        parser.add_argument('--cmd-vel-x-window', type=float, nargs=2, metavar=('MIN', 'MAX'), default=CMD_VEL_X_WINDOW)
        parser.add_argument('--ood-feature', choices=('cmd_vel', 'environment'), default=OOD_FEATURE)
        parser.add_argument('--environment-window', type=int, nargs=2, action='append', metavar=('MIN', 'MAX'), default=ENVIRONMENT_WINDOWS)
        parser.add_argument('--first-seed', type=int, default=FIRST_SEED)
        parser.add_argument('--last-seed', type=int, default=LAST_SEED)
        parser.add_argument('--output-csv', type=Path)
        parser.add_argument('--with-umap', action='store_true')
        parser.add_argument('--skip-knn', action='store_true', default=not RUN_KNN)
        parser.add_argument('--plot-metric', choices=('epistemic', 'negative-log-density'), default=PLOT_METRIC)
        return parser

    @staticmethod
    def row_from_metrics(metrics):
        scientific = lambda value: '' if value is None else f'{float(value):.4e}'
        vector = lambda name: metrics.get(name, (None, None, None))
        return {
            'seed': metrics.get('seed', ''), 'ood_feature': metrics.get('ood_feature', ''),
            'environment': metrics.get('environment', ''), 'cmd_vel_x': scientific(metrics.get('cmd_vel_x')),
            'velocity_mae': scientific(metrics.get('velocity_mae')),
            'uncertainty_final_timestep_gt_contact': metrics.get('uncertainty_final_timestep_gt_contact', ''),
            **{f'aleatoric_v{axis}': scientific(value) for axis, value in zip('xyz', vector('aleatoric_variance'))},
            **{f'epistemic_v{axis}': scientific(value) for axis, value in zip('xyz', vector('epistemic_variance'))},
            **{f'total_v{axis}': scientific(value) for axis, value in zip('xyz', vector('total_variance'))},
            'negative_log_density': scientific(metrics.get('negative_log_density')),
            'knn_ood_windows': metrics.get('knn_ood_windows', ''), 'knn_total_windows': metrics.get('knn_total_windows', ''),
            'knn_ood_percentage_total_windows': scientific(metrics.get('knn_ood_percentage_total_windows')),
        }

    @staticmethod
    def create_in_process_runner(args, root, knn_cache_path):
        sys.path.insert(0, str(root / 'src'))
        if args.model == 'natpn':
            from testNatPN import NatPNSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = NatPNSingleTest(), None, None
        elif args.model == 'der':
            from testDER import DERSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = DERSingleTest(), None, None
        elif args.model == 'ensemble':
            from testEnsemble import EnsembleSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = EnsembleSingleTest(), None, None
        else:
            raise ValueError(f'Unknown model {args.model}')

        def run_seed(seed):
            nonlocal cached_model, cached_checkpoint
            single_args = Namespace(
                config_name=str(args.config_name), seed=seed, cmd_vel_x_window=args.cmd_vel_x_window,
                ood_feature=args.ood_feature, environment_window=args.environment_window, metrics_json=None,
                skip_umap=not args.with_umap, save_umap=args.with_umap, skip_knn=args.skip_knn,
                knn_cache=str(knn_cache_path), skip_plots=True,
            )
            context = evaluator.prepare(single_args)
            if args.model == 'ensemble':
                metrics, cached_model, cached_checkpoint = run_evaluation(
                    single_args, context, models=cached_model, checkpoint_paths=cached_checkpoint
                )
            else:
                metrics, cached_model, cached_checkpoint = run_evaluation(
                    single_args, context, model=cached_model, checkpoint_path=cached_checkpoint
                )
            return metrics

        return run_seed

    def main(self, argv=None):
        root = Path(__file__).resolve().parents[1]
        args = self.build_parser(root).parse_args(argv)
        if args.first_seed > args.last_seed:
            raise ValueError('--first-seed must be no greater than --last-seed')
        if args.output_csv is None:
            args.output_csv = root / 'testResults' / f'{args.model}_seeds_{args.first_seed}-{args.last_seed}.csv'
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.create_in_process_runner(args, root, Path(temp_dir) / 'knn_reference.npz')
            for seed in range(args.first_seed, args.last_seed + 1):
                print(f'Running seed {seed}')
                rows.append(self.row_from_metrics(runner(seed)))

        with open(args.output_csv, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows)} rows to {args.output_csv}')
        subprocess.run([
            sys.executable, root / 'tests/make_testsingle_table.py', '--input-csv', args.output_csv,
            '--plot-metric', args.plot_metric,
        ], cwd=root, check=True)


if __name__ == '__main__':
    SeriesTest().main()
