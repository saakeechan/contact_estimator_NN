"""Render a seed-sweep CSV as a paginated PDF results table."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize


def number(value, digits=4):
    value = float(value)
    return f'{value:.{digits}f}' if 1e-3 <= abs(value) < 1e4 else f'{value:.{digits - 2}e}'


def vector(row, prefix):
    return '[' + ', '.join(number(row[f'{prefix}_v{axis}']) for axis in 'xyz') + ']'


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path)
    parser.add_argument('--output-pdf', type=Path)
    parser.add_argument('--output-plot', type=Path)
    parser.add_argument('--save-pdf', action='store_true', help='Write the PDF table.')
    parser.add_argument('--save-plot', action='store_true', help='Write the PNG plot.')
    parser.add_argument('--plot-metric', choices=('epistemic', 'negative-log-density', 'velocity-mae'), default='epistemic')
    args = parser.parse_args()

    input_csv = args.input_csv or max((root / 'logs').glob('testsingle_seed_sweep_*.csv'), key=lambda path: path.stat().st_mtime)
    output_pdf = args.output_pdf or input_csv.with_suffix('.pdf')
    with open(input_csv, newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f'No rows in {input_csv}')
    ood_feature = rows[0].get('ood_feature', 'cmd_vel')
    if any(row.get('ood_feature', ood_feature) != ood_feature for row in rows):
        raise ValueError('All rows must use the same ood_feature to create one comparison plot.')
    if ood_feature not in {'cmd_vel', 'environment'}:
        raise ValueError(f'Unsupported ood_feature: {ood_feature}')
    feature_label = 'Environment' if ood_feature == 'environment' else 'Cmd vel'
    feature_column = 'environment' if ood_feature == 'environment' else 'cmd_vel_x'
    output_suffix = 'environment' if ood_feature == 'environment' else 'cmd_vel'
    metric_label = {'epistemic': 'Mean epistemic variance', 'negative-log-density': 'Negative log density',
                    'velocity-mae': 'Contact-masked velocity MAE'}[args.plot_metric]
    metric_suffix = {'epistemic': 'mean_epistemic', 'negative-log-density': 'negative_log_density',
                     'velocity-mae': 'velocity_mae_vs_cmd_vel'}[args.plot_metric]
    output_plot = args.output_plot or input_csv.with_name(f'{input_csv.stem}_{metric_suffix}_vs_{output_suffix}.png')

    headers = ['Seed', feature_label, 'MAE', 'GT contact', 'Aleatoric variance [vx, vy, vz]',
               'Epistemic variance [vx, vy, vz]', 'kNN OOD']
    table_rows = []
    for row in rows:
        ood = 'Not run' if row['knn_ood_windows'] == '' else (
            f"{row['knn_ood_windows']}/{row['knn_total_windows']} = "
            f"{100 * float(row['knn_ood_percentage_total_windows']):.2f}%"
        )
        table_rows.append([row['seed'], number(row[feature_column]), number(row['velocity_mae'], 5),
                           row['uncertainty_final_timestep_gt_contact'], vector(row, 'aleatoric'),
                           vector(row, 'epistemic'), ood])

    if args.save_pdf:
        with PdfPages(output_pdf) as pdf:
            for first in range(0, len(table_rows), 16):
                figure, axis = plt.subplots(figsize=(16, 8.5))
                axis.axis('off')
                table = axis.table(
                    cellText=table_rows[first:first + 16], colLabels=headers, loc='center',
                    colWidths=[0.05, 0.08, 0.08, 0.07, 0.25, 0.25, 0.14],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 1.65)
                figure.suptitle(f'{input_csv.stem} ({first + 1}-{min(first + 16, len(table_rows))} of {len(table_rows)})')
                figure.tight_layout()
                pdf.savefig(figure, bbox_inches='tight')
                plt.close(figure)

    feature_values = np.asarray([float(row[feature_column]) for row in rows])
    plot_values = np.asarray(
        [sum(float(row[f'epistemic_v{axis}']) for axis in 'xyz') / 3 for row in rows]
        if args.plot_metric == 'epistemic' else
        [float(row['negative_log_density']) for row in rows] if args.plot_metric == 'negative-log-density' else
        [float(row['velocity_mae']) for row in rows]
    )
    velocity_mae = np.asarray([float(row['velocity_mae']) for row in rows])
    finite_mae = velocity_mae[np.isfinite(velocity_mae)]
    finite_plot = plot_values[np.isfinite(plot_values)]
    mae_max = np.percentile(finite_mae, 99)
    y_min, y_max = finite_plot.min(), finite_plot.max()  # Temporary: show the full epistemic range.
    y_max = max(y_max, y_min * (1.01 if args.plot_metric == 'epistemic' else 1.0) + 1e-12)

    if args.save_plot:
        figure, axis = plt.subplots(figsize=(9, 6))
        if args.plot_metric == 'velocity-mae':
            axis.scatter(feature_values, velocity_mae, color='tab:blue', s=24)
            x_label = 'Environment number' if ood_feature == 'environment' else 'Command velocity x (m/s)'
            axis.set(xlabel=x_label, ylabel=metric_label)
        else:
            points = axis.scatter(
                feature_values, plot_values, c=velocity_mae, cmap='viridis',
                norm=Normalize(vmin=finite_mae.min(), vmax=mae_max, clip=True),
            )
            figure.colorbar(points, ax=axis, extend='max', label='Contact-masked velocity MAE (95th-percentile cap)')
            x_label = 'Environment number' if ood_feature == 'environment' else 'Command velocity x (m/s)'
            axis.set(xlabel=x_label, ylabel=metric_label, yscale='log' if args.plot_metric == 'epistemic' else 'linear')
            axis.set_ylim(y_min, y_max)
        axis.grid(True, which='both', alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_plot, dpi=150)
        plt.close(figure)
    if args.save_pdf:
        print(f'Wrote {output_pdf}')
    if args.save_plot:
        print(f'Wrote {output_plot}')


if __name__ == '__main__':
    main()
