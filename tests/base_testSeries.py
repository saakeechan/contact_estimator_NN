"""Run one configurable NatPN, DER, or deep-ensemble evaluation seed sweep."""
import argparse
import csv
from argparse import Namespace
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

# Edit these defaults, or override them on the command line.
MODEL = 'mc_dropout'  # 'natpn', 'der', 'ensemble', or 'mc_dropout'
FIRST_SEED = 102
LAST_SEED = 300
OOD_FEATURE = 'cmd_vel'
CMD_VEL_X_WINDOW = (0.0, 3.0)
ENVIRONMENT_WINDOWS = [(0, 20)]
PLOT_METRIC = 'epistemic'
RUN_KNN = False
SAVE_CSV = False
SAVE_PDF = False
SAVE_PNG = True

MODEL_SCRIPTS = {
    'natpn': 'tests/testNatPN.py',
    'der': 'tests/testDER.py',
    'ensemble': 'tests/testEnsemble.py',
    'mc_dropout': 'tests/testMCdropout.py',
    'ucb': 'tests/testUCB.py',
    'vcl': 'tests/testVCL.py',
    'replay': 'tests/testReplay.py',
    'replay_mc_dropout': 'tests/testReplayMCdropout.py',
}
MODEL_RESULTS_DIRECTORIES = {
    'natpn': 'NatPN',
    'der': 'DER',
    'ensemble': 'Ensemble',
    'mc_dropout': 'MCDropout',
    'ucb': 'UCB',
    'vcl': 'VCL',
    'replay': 'Replay',
    'replay_mc_dropout': 'ReplayMCDropout',
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
        parser.add_argument('--save-csv', action='store_true', help='Write --output-csv even when SAVE_CSV is false.')
        parser.add_argument('--fail-if-output-exists', action='store_true',
                            help='Refuse to replace an existing requested result artifact.')
        parser.add_argument('--checkpoint-path', type=Path,
                            help='Evaluate this exact checkpoint instead of discovering the newest one.')
        parser.add_argument('--with-umap', action='store_true')
        parser.add_argument('--skip-knn', action='store_true', default=not RUN_KNN)
        parser.add_argument('--plot-metric', choices=('epistemic', 'negative-log-density', 'velocity-mae'), default=PLOT_METRIC)
        parser.add_argument('--quiet', action='store_true', help='Suppress per-seed trajectory output.')
        return parser

    @staticmethod
    def row_from_metrics(metrics):
        scientific = lambda value: '' if value is None else f'{float(value):.4e}'
        vector = lambda name: metrics.get(name, (None, None, None))
        return {
            'seed': metrics.get('seed', ''), 'ood_feature': metrics.get('ood_feature', ''),
            'environment': metrics.get('environment', ''), 'run_index': metrics.get('run_index', ''),
            'cmd_vel_x': scientific(metrics.get('cmd_vel_x')),
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
        if args.model == 'natpn':
            from tests.testNatPN import NatPNSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = NatPNSingleTest(), None, None
        elif args.model == 'der':
            from tests.testDER import DERSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = DERSingleTest(), None, None
        elif args.model == 'ensemble':
            from tests.testEnsemble import EnsembleSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = EnsembleSingleTest(), None, None
        elif args.model == 'mc_dropout':
            from tests.testMCdropout import MCDropoutSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = MCDropoutSingleTest(), None, None
        elif args.model == 'ucb':
            from tests.testUCB import UCBSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = UCBSingleTest(), None, None
        elif args.model == 'vcl':
            from tests.testVCL import VCLSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = VCLSingleTest(), None, None
        elif args.model == 'replay':
            from tests.testReplay import ReplaySingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = ReplaySingleTest(), None, None
        elif args.model == 'replay_mc_dropout':
            from tests.testReplayMCdropout import ReplayMCDropoutSingleTest, run_evaluation
            evaluator, cached_model, cached_checkpoint = ReplayMCDropoutSingleTest(), None, None
        else:
            raise ValueError(f'Unknown model {args.model}')

        def run_seed(seed):
            nonlocal cached_model, cached_checkpoint
            single_args = Namespace(
                config_name=str(args.config_name), seed=seed, cmd_vel_x_window=args.cmd_vel_x_window,
                ood_feature=args.ood_feature, environment_window=args.environment_window, metrics_json=None,
                skip_umap=not args.with_umap, save_umap=args.with_umap, skip_knn=args.skip_knn,
                knn_cache=str(knn_cache_path), skip_plots=True,
                mc_samples=None, quiet=args.quiet,
            )
            context = evaluator.prepare(single_args)
            if args.model == 'ensemble':
                metrics, cached_model, cached_checkpoint = run_evaluation(
                    single_args, context, models=cached_model, checkpoint_paths=cached_checkpoint
                )
            else:
                metrics, cached_model, cached_checkpoint = run_evaluation(
                    single_args, context, model=cached_model,
                    checkpoint_path=args.checkpoint_path or cached_checkpoint,
                )
            return metrics

        return run_seed

    def main(self, argv=None):
        root = Path(__file__).resolve().parents[1]
        args = self.build_parser(root).parse_args(argv)
        if args.first_seed > args.last_seed:
            raise ValueError('--first-seed must be no greater than --last-seed')
        if args.output_csv is None:
            args.output_csv = (
                root / 'testResults' / MODEL_RESULTS_DIRECTORIES[args.model]
                / f'{args.model}_seeds_{args.first_seed}-{args.last_seed}.csv'
            )
        plot_suffix = {'epistemic': 'mean_epistemic', 'negative-log-density': 'negative_log_density',
                       'velocity-mae': 'velocity_mae_vs_cmd_vel'}[args.plot_metric]
        output_paths = []
        save_csv = SAVE_CSV or args.save_csv
        if save_csv:
            output_paths.append(args.output_csv)
        if SAVE_PDF:
            output_paths.append(args.output_csv.with_suffix('.pdf'))
        if SAVE_PNG:
            output_paths.append(args.output_csv.with_name(
                f'{args.output_csv.stem}_{plot_suffix}_vs_{args.ood_feature}.png'
            ))
        if args.fail_if_output_exists:
            existing_outputs = [path for path in output_paths if path.exists()]
            if existing_outputs:
                raise FileExistsError(f'Refusing to overwrite existing result artifact(s): {existing_outputs}')

        rows = []
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.create_in_process_runner(args, root, Path(temp_dir) / 'knn_reference.npz')
            for seed in range(args.first_seed, args.last_seed + 1):
                if not args.quiet:
                    print(f'Running seed {seed}')
                rows.append(self.row_from_metrics(runner(seed)))

        if not any((save_csv, SAVE_PDF, SAVE_PNG)):
            print('Results were not saved (SAVE_CSV, SAVE_PDF, and SAVE_PNG are all False).')
            return

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        def write_csv(path):
            with open(path, 'w', newline='') as output_file:
                writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        if save_csv:
            write_csv(args.output_csv)
            print(f'Wrote {len(rows)} rows to {args.output_csv}')
        if not (SAVE_PDF or SAVE_PNG):
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv = args.output_csv if save_csv else Path(temp_dir) / args.output_csv.name
            if not save_csv:
                write_csv(input_csv)
            command = [
                sys.executable, root / 'tests/make_testsingle_table.py', '--input-csv', input_csv,
                '--plot-metric', args.plot_metric,
            ]
            if SAVE_PDF:
                command.extend(('--save-pdf', '--output-pdf', args.output_csv.with_suffix('.pdf')))
            if SAVE_PNG:
                command.extend(('--save-plot', '--output-plot', args.output_csv.with_name(
                    f'{args.output_csv.stem}_{plot_suffix}_vs_{args.ood_feature}.png'
                )))
            subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    SeriesTest().main()
