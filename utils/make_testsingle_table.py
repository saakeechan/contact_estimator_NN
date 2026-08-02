"""Render a seed-sweep CSV as a paginated PDF results table."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
    args = parser.parse_args()

    input_csv = args.input_csv or max((root / 'logs').glob('testsingle_seed_sweep_*.csv'), key=lambda path: path.stat().st_mtime)
    output_pdf = args.output_pdf or input_csv.with_suffix('.pdf')
    output_plot = args.output_plot or input_csv.with_name(f'{input_csv.stem}_mean_epistemic_vs_cmd_vel.png')
    with open(input_csv, newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f'No rows in {input_csv}')

    headers = ['Seed', 'Cmd vel', 'MAE', 'GT contact', 'Aleatoric variance [vx, vy, vz]',
               'Epistemic variance [vx, vy, vz]', 'kNN OOD']
    table_rows = []
    for row in rows:
        ood = f"{row['knn_ood_windows']}/{row['knn_total_windows']} = {100 * float(row['knn_ood_percentage_total_windows']):.2f}%"
        table_rows.append([row['seed'], number(row['cmd_vel_x']), number(row['velocity_mae'], 5),
                           row['uncertainty_final_timestep_gt_contact'], vector(row, 'aleatoric'),
                           vector(row, 'epistemic'), ood])

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

    cmd_velocity = [float(row['cmd_vel_x']) for row in rows]
    mean_epistemic = [sum(float(row[f'epistemic_v{axis}']) for axis in 'xyz') / 3 for row in rows]
    velocity_mae = [float(row['velocity_mae']) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 6))
    points = axis.scatter(cmd_velocity, mean_epistemic, c=velocity_mae, cmap='viridis')
    figure.colorbar(points, ax=axis, label='Contact-masked velocity MAE')
    axis.set(xlabel='Command velocity x (m/s)', ylabel='Mean epistemic variance', yscale='log')
    axis.grid(True, which='both', alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_plot, dpi=150)
    plt.close(figure)
    print(f'Wrote {output_pdf}')
    print(f'Wrote {output_plot}')


if __name__ == '__main__':
    main()
