"""Run testSingle.py over a seed range and collect its metrics in a CSV."""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', default=root / 'config/network_params.yaml', type=Path)
    parser.add_argument('--cmd-vel-x-window', type=float, nargs=2, metavar=('MIN', 'MAX'), default=(1.5, 3.0))
    parser.add_argument('--first-seed', type=int, default=201)
    parser.add_argument('--last-seed', type=int, default=300)
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--with-umap', action='store_true', help='Also save one UMAP figure per seed.')
    args = parser.parse_args()
    if args.first_seed > args.last_seed:
        parser.error('--first-seed must be no greater than --last-seed')
    if args.output_csv is None:
        args.output_csv = root / 'testResults' / f'der_seeds_{args.first_seed}-{args.last_seed}.csv'

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scientific = lambda value: f'{float(value):.4e}'
    rows = []
    with tempfile.TemporaryDirectory() as temp_dir:
        knn_cache_path = Path(temp_dir) / 'knn_reference.npz'
        for seed in range(args.first_seed, args.last_seed + 1):
            metrics_path = Path(temp_dir) / f'{seed}.json'
            command = [
                sys.executable, root / 'src/testSingleDER.py',
                '--config_name', args.config_name, '--seed', str(seed),
                '--cmd-vel-x-window', *(str(value) for value in args.cmd_vel_x_window),
                '--metrics-json', metrics_path,
                '--knn-cache', knn_cache_path,
                '--skip-plots',
            ]
            if args.with_umap:
                command.append('--save-umap')
            else:
                command.append('--skip-umap')
            print(f'Running seed {seed}')
            subprocess.run(command, cwd=root, check=True)
            with open(metrics_path) as metrics_file:
                metrics = json.load(metrics_file)
            rows.append({
                'seed': metrics['seed'], 'cmd_vel_x': scientific(metrics['cmd_vel_x']),
                'velocity_mae': scientific(metrics['velocity_mae']),
                'uncertainty_final_timestep_gt_contact': int(metrics['uncertainty_final_timestep_gt_contact']),
                **{f'aleatoric_v{axis}': scientific(value) for axis, value in zip('xyz', metrics['aleatoric_variance'])},
                **{f'epistemic_v{axis}': scientific(value) for axis, value in zip('xyz', metrics['epistemic_variance'])},
                'knn_ood_windows': metrics['knn_ood_windows'],
                'knn_total_windows': metrics['knn_total_windows'],
                'knn_ood_percentage_total_windows': scientific(metrics['knn_ood_percentage_total_windows']),
            })

    with open(args.output_csv, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output_csv}')
    subprocess.run(
        [sys.executable, root / 'utils/make_testsingle_table.py', '--input-csv', args.output_csv],
        cwd=root, check=True,
    )


if __name__ == '__main__':
    main()
