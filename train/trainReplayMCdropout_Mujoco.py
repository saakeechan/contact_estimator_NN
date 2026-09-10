"""Continual MC-dropout replay over ordered MuJoCo environments."""
import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / 'src')]

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mc_dropout import MCDropoutTCN
from normalization import ContactCNNWithNormalization
from tests.base_testSeries_crossSim import (
    SCENARIO_FOLDERS, SCENARIO_STYLES, load_dataset, run_cross_sim_evaluation, validate_compatible_datasets,
)
from train.trainMCdropout import MCDropoutTrainer
from train.trainReplay import _balanced_replay, _load_config, _normalization_stats


class EnvironmentWindows(Dataset):
    def __init__(self, datasets, references, window_size):
        self.datasets, self.references, self.window_size = datasets, np.asarray(references, dtype=np.int64), int(window_size)
        if not len(self.references):
            raise ValueError('A MuJoCo task has no valid windows.')

    def __len__(self): return len(self.references)

    def __getitem__(self, index):
        environment_id, start = self.references[index]
        dataset = self.datasets[int(environment_id)]
        end = int(start) + self.window_size
        tensor = lambda values: torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
        return {'data': tensor(dataset['data'][start:end]),
                'label': tensor(dataset['labels'][end - 1]),
                'label_seq': tensor(dataset['labels'][start:end]),
                'velocity': tensor(dataset['velocities'][start:end])}


def _task_references(dataset, environment_id, window_size):
    starts, run_ids, offset = [], [], 0
    for run_id, end in enumerate(dataset['boundaries']):
        run_starts = np.arange(offset, int(end) - window_size + 1, dtype=np.int64)
        if len(run_starts):
            starts.append(run_starts); run_ids.append(np.full(len(run_starts), run_id, dtype=np.int64))
        offset = int(end)
    if not starts:
        raise ValueError(f"{dataset['name']} has no {window_size}-sample windows.")
    return np.column_stack((np.full(sum(map(len, starts)), environment_id), np.concatenate(starts))), np.concatenate(run_ids)


def _split(references, run_ids, train_ratio, seed, no_validation=False):
    if no_validation:
        return references, None
    runs = np.unique(run_ids)
    if len(runs) == 1:  # No run-disjoint validation exists; train on the full environment.
        return references, references
    rng = np.random.default_rng(seed); rng.shuffle(runs)
    train_runs = runs[:min(len(runs) - 1, max(1, int(train_ratio * len(runs))))]
    return references[np.isin(run_ids, train_runs)], references[~np.isin(run_ids, train_runs)]


def _loader(datasets, references, config, shuffle, seed):
    return DataLoader(EnvironmentWindows(datasets, references, config['window_size']), batch_size=config['batch_size'],
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed) if shuffle else None)


def _save_historical_metrics(rows, environments, output_path):
    """Plot each environment's mean evaluation metrics after every task."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, (epistemic_axis, mae_axis) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for environment in environments:
        environment_rows = [row for row in rows if row['simulator'] == environment]
        task_ids = sorted({row['checkpoint_after_task'] for row in environment_rows})
        epistemic = [np.mean([row['log10_mean_epistemic_variance'] for row in environment_rows
                              if row['checkpoint_after_task'] == task_id]) for task_id in task_ids]
        mae = [np.mean([row['velocity_mae'] for row in environment_rows
                        if row['checkpoint_after_task'] == task_id]) for task_id in task_ids]
        style = SCENARIO_STYLES[environment]
        epistemic_axis.plot(task_ids, epistemic, label=f'{environment} (log epi)', **style)
        mae_axis.plot(task_ids, mae, label=f'{environment} (MAE)', **style)
    epistemic_axis.set(ylabel='log Epistemic variance')
    mae_axis.set(xlabel='Training task completed', ylabel='Velocity Error')
    mae_axis.set_xticks(range(len(environments)), [f'Task {task_id}\n{name}' for task_id, name in enumerate(environments)])
    for axis in (epistemic_axis, mae_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_epistemic_error_correlation(rows, environments, output_path):
    """Plot all evaluated windows and their pooled Pearson/Spearman correlations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log_epistemic = np.asarray([row['log10_mean_epistemic_variance'] for row in rows])
    mae = np.asarray([row['velocity_mae'] for row in rows])
    correlation = (np.corrcoef(log_epistemic, mae)[0, 1]
                   if len(rows) > 1 and np.ptp(log_epistemic) and np.ptp(mae) else np.nan)
    def average_ranks(values):
        order = np.argsort(values, kind='stable')
        ranks = np.empty(len(values), dtype=float)
        boundaries = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1, len(values)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            ranks[order[start:end]] = (start + end + 1) / 2
        return ranks
    spearman = (np.corrcoef(average_ranks(log_epistemic), average_ranks(mae))[0, 1]
                if len(rows) > 1 and np.ptp(log_epistemic) and np.ptp(mae) else np.nan)
    figure, axis = plt.subplots(figsize=(9, 6))
    for environment in environments:
        environment_rows = [row for row in rows if row['simulator'] == environment]
        environment_epistemic = [row['log10_mean_epistemic_variance'] for row in environment_rows]
        environment_mae = [row['velocity_mae'] for row in environment_rows]
        axis.scatter(environment_epistemic, environment_mae, s=30,
                     label=f'{environment} (log epi={np.mean(environment_epistemic):.2f}, MAE={np.mean(environment_mae):.2e})',
                     **SCENARIO_STYLES[environment])
    axis.set(xlabel='log Epistemic variance', ylabel='Velocity Error',
             title=f'Pearson r = {correlation:.3f}; Spearman ρ = {spearman:.3f} (all environments)')
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', type=Path, default=_ROOT / 'config/network_params.yaml')
    parser.add_argument('--replay-mc-config', type=Path, default=_ROOT / 'config/ReplayMCdropoutMujoco_params.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-evaluate-after-task', action='store_true')
    args = parser.parse_args()
    config = _load_config(args.config_name, args.replay_mc_config)
    environments = tuple(config.get('continual_environments', ()))
    if not environments or len(set(environments)) != len(environments):
        raise ValueError('continual_environments must be a non-empty list without duplicates.')
    if set(environments) - set(SCENARIO_FOLDERS):
        raise ValueError(f'Unknown MuJoCo environment(s): {sorted(set(environments) - set(SCENARIO_FOLDERS))}')
    datasets = [load_dataset(name, SCENARIO_FOLDERS[name]) for name in environments]
    reference = validate_compatible_datasets(datasets)
    config['data_folder'], config['legs'] = str(reference['folder']), tuple(reference['metadata']['legs'])
    if not 0 < float(config['mc_dropout_rate']) < 1 or int(config['mc_dropout_samples']) < 2:
        raise ValueError('mc_dropout_rate must be in (0, 1) and mc_dropout_samples must be at least 2.')
    seed = int(config.get('random_seed', 42))
    task_refs, splits = [], []
    no_validation = config.get('val_ratio') is None or str(config.get('val_ratio')).lower() == 'none'
    for task_id, dataset in enumerate(datasets):
        refs, run_ids = _task_references(dataset, task_id, config['window_size'])
        task_refs.append(refs); splits.append(_split(refs, run_ids, float(config.get('train_ratio', .85)), seed + task_id, no_validation))
        print(f'Task {task_id} {dataset["name"]}: {len(splits[-1][0])} train' + ('' if splits[-1][1] is None else f', {len(splits[-1][1])} validation windows'))
    if args.dry_run: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = _normalization_stats(datasets[0]['data'], splits[0][0][:, 1], config['window_size'], device)
    model = ContactCNNWithNormalization(MCDropoutTCN(config['window_size'], reference['data'].shape[1], config.get('tcn_num_channels', 64), config.get('tcn_kernel_size', 3), config.get('tcn_num_blocks', 5), config['mc_dropout_rate'], legs=config['legs']), mean, std).to(device)
    run_dir = _ROOT / 'logs' / 'logsReplayMCDropoutMujoco' / f'run_{datetime.now():%Y-%m-%d_%H-%M-%S}'
    run_dir.mkdir(parents=True)
    rows, evaluation_rows, completed, replay = [], [], [], np.empty((0, 2), dtype=np.int64)
    for task_id, (train_refs, val_refs) in enumerate(splits):
        trainer = MCDropoutTrainer(model, config.copy(), str(run_dir / f'task_{task_id:02d}'))
        trainer.train(_loader(datasets, np.concatenate((train_refs, replay)), config, True, seed + task_id), None if val_refs is None else _loader(datasets, val_refs, config, False, seed))
        state = torch.load(run_dir / f'task_{task_id:02d}' / 'model_final_epoch.pt', map_location=device)
        model.load_state_dict(state['model_state_dict']); completed.append(train_refs)
        replay = _balanced_replay(completed, int(config['replay_capacity']), int(config.get('replay_seed', seed)))
        checkpoint = run_dir / f'model_after_task_{task_id:02d}.pt'
        torch.save({'model_state_dict': model.state_dict(), 'task_id': task_id, 'replay_references': replay.tolist(), 'task_train_references': [x.tolist() for x in completed], 'config': config}, checkpoint)
        if config.get('replay_evaluate_after_task', True) and not args.skip_evaluate_after_task:
            output = _ROOT / 'testResults' / 'ReplayMCDropoutMujoco' / run_dir.name / f'series_after_task_{task_id:02d}.csv'
            task_evaluation_rows = run_cross_sim_evaluation(
                'replay_mc_dropout_mujoco', args.config_name, environments, checkpoint,
                int(config['replay_evaluation_first_seed']), int(config['replay_evaluation_last_seed']), output,
                False, True, cmd_vel_x_window=config['evaluation_window'])
            evaluation_rows.extend({**row, 'checkpoint_after_task': task_id} for row in task_evaluation_rows)
            correlation_output = output.with_name(f'epistemic_mae_correlation_after_task_{task_id:02d}.png')
            _save_epistemic_error_correlation(task_evaluation_rows, environments, correlation_output)
            print(f'Wrote {correlation_output}')
        if not no_validation:
            for evaluated_task, (_, evaluation_refs) in enumerate(splits[:task_id + 1]):
                metrics = trainer.evaluate(_loader(datasets, evaluation_refs, config, False, seed))
                rows.append({'checkpoint_after_task': task_id, 'evaluation_environment': environments[evaluated_task], **metrics})
            with (run_dir / 'task_metrics.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    if evaluation_rows:
        output = _ROOT / 'testResults' / 'ReplayMCDropoutMujoco' / run_dir.name / 'historical_metrics.png'
        _save_historical_metrics(evaluation_rows, environments, output)
        print(f'Wrote {output}')


if __name__ == '__main__': main()
