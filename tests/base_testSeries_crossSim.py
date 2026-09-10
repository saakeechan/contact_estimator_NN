"""Compare one uncertainty-aware model across selected Isaac Sim/MuJoCo scenarios.

The model is loaded exactly once. For every seed, the runner evaluates one
contact-final, boundary-safe window from every scenario in ``TEST_SCENARIOS``
and overlays them against that window's command velocity.
"""
import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

import numpy as np
import torch


# Edit these values for a cross-simulation run.  MODEL chooses the saved model
# implementation; CHECKPOINT_PATH=None selects that implementation's newest
# compatible checkpoint.
MODEL = 'mc_dropout'  # 'mc_dropout', 'ucb', 'vcl', or 'ensemble'
CHECKPOINT_PATH = None
ISAAC_DATA_FOLDER = _ROOT / 'Data' / 'NumpyFiles'
MUJOCO_DATA_FOLDER = _ROOT / 'Data' / 'MujocoNumpyFiles'
# Edit this list to choose the test conditions, e.g.
# ('Dampy', 'Flat', 'Slope', 'Isaac'). MuJoCo conditions are read from their
# own subfolders under Data/MujocoNumpyFiles.
TEST_SCENARIOS = ('Slope', 'Flat2')
SCENARIO_FOLDERS = {
    'Dampy': MUJOCO_DATA_FOLDER / 'Dampy',
    'Flat': MUJOCO_DATA_FOLDER / 'Flat',
    'Flat2': MUJOCO_DATA_FOLDER / 'Flat2',
    'Slope': MUJOCO_DATA_FOLDER / 'Slope',
    'Backwards': MUJOCO_DATA_FOLDER / 'Backwards',
    'Isaac': ISAAC_DATA_FOLDER,
    'Sideways': MUJOCO_DATA_FOLDER / 'Sideways',
    'Slip': MUJOCO_DATA_FOLDER / 'Slip'
}
SCENARIO_STYLES = {
    'Isaac': {'color': 'tab:blue', 'marker': 'o'},
    'Flat': {'color': 'tab:orange', 'marker': 's'},
    'Dampy': {'color': 'tab:green', 'marker': '^'},
    'Slope': {'color': 'tab:red', 'marker': 'D'},
    'Backwards': {'color': 'tab:purple', 'marker': 'v'},
    'Sideways': {'color': 'tab:brown', 'marker': 'p'},
    'Slip': {'color': 'tab:gray', 'marker': '8'},
    'Flat2': {'color': 'tab:olive', 'marker': 'x'}

}

FIRST_SEED = 1100
LAST_SEED = 1200
SAVE_CSV = False
SAVE_PNG = True

MODEL_EVALUATORS = {
    'mc_dropout': ('tests.testMCdropout', 'MCDropoutSingleTest'),
    'ucb': ('tests.testUCB', 'UCBSingleTest'),
    'vcl': ('tests.testVCL', 'VCLSingleTest'),
    'ensemble': ('tests.testEnsemble', 'EnsembleSingleTest'),
    'replay_mc_dropout_mujoco': ('tests.testReplayMCdropout_Mujoco', 'ReplayMCDropoutMujocoSingleTest'),
}


def load_dataset(name, folder):
    """Load the three canonical arrays and validate their shared sample axis."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'{name} dataset folder does not exist: {folder}')
    data = np.load(folder / 'all_data.npy', mmap_mode='r')
    labels = np.load(folder / 'all_labels.npy', mmap_mode='r')
    velocities = np.load(folder / 'all_body_velocities.npy', mmap_mode='r')
    boundaries = np.load(folder / 'all_data_boundaries.npy')
    metadata = np.load(folder / 'all_data_metadata.npy', allow_pickle=True).item()
    if not (len(data) == len(labels) == len(velocities)):
        raise ValueError(f'{folder}: data, labels, and velocities have inconsistent lengths.')
    if int(metadata['num_features']) != data.shape[1]:
        raise ValueError(f'{folder}: metadata num_features does not match all_data.npy.')
    return {
        'name': name,
        'folder': folder, 'data': data, 'labels': labels, 'velocities': velocities,
        'boundaries': boundaries, 'metadata': metadata,
    }


def valid_window_starts(boundaries, window_size):
    """Return starts whose complete temporal window remains inside one run."""
    starts, run_start = [], 0
    for run_end in boundaries:
        run_end = int(run_end)
        if run_end - run_start >= window_size:
            starts.append(np.arange(run_start, run_end - window_size + 1, dtype=np.int64))
        run_start = run_end
    if not starts:
        raise ValueError(f'No runs contain the requested {window_size}-sample window.')
    return np.concatenate(starts)


def load_evaluator_and_model(model_name, config_name, num_features, legs, checkpoint_path):
    """Build the selected implementation once, using its normal checkpoint lookup."""
    if model_name not in MODEL_EVALUATORS:
        raise ValueError(f'Unsupported MODEL={model_name!r}; choose from {tuple(MODEL_EVALUATORS)}.')
    module_name, class_name = MODEL_EVALUATORS[model_name]
    module = __import__(module_name, fromlist=[class_name])
    evaluator = getattr(module, class_name)()
    # These model evaluators use this optional attribute while loading config.
    evaluator.set_runtime_values(argparse.Namespace(mc_samples=None))
    config = evaluator.load_config(config_name)
    config['legs'] = tuple(legs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'ensemble':
        checkpoint_paths = evaluator.find_member_checkpoints(config, num_features)
        models = []
        for path in checkpoint_paths:
            model = evaluator.build_member(config, num_features).to(device)
            model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
            models.append(model.eval())
        return evaluator, config, device, models, checkpoint_paths

    model = evaluator.build_model(config, num_features).to(device)
    path = Path(checkpoint_path) if checkpoint_path else Path(evaluator.find_checkpoint(num_features, config))
    state_dict = torch.load(path, map_location=device)['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Older UCB checkpoints deliberately lack newly registered prior buffers.
        if model_name != 'ucb':
            raise
        getattr(module, '_load_ucb_inference_checkpoint')(model, state_dict)
    return evaluator, config, device, model.eval(), path


@torch.no_grad()
def predict_with_epistemic(model_name, evaluator, model, windows, samples):
    """Return predictive mean and MC/member disagreement for an input batch."""
    if model_name == 'ensemble':
        outputs = [member(windows, return_sequence=False) for member in model]
        member_means = torch.stack([output[1] for output in outputs])
        return member_means.mean(dim=0), member_means.var(dim=0, unbiased=False)
    mean, _, epistemic = evaluator.stochastic_outputs(model, windows, samples)
    return mean, epistemic


def evaluate_dataset(model_name, evaluator, config, device, model, dataset, seed, scenario_index):
    """Evaluate one reproducibly selected contact-final window."""
    starts = valid_window_starts(dataset['boundaries'], config['window_size'])
    final_indices = starts + config['window_size'] - 1
    contacts = np.asarray(dataset['labels'][final_indices])
    if contacts.ndim == 1:
        contacts = contacts[:, None]
    starts = starts[(contacts == 1).any(axis=1)]
    if not len(starts):
        raise ValueError(f"{dataset['name']} has no contact-final windows.")
    selected_start = int(np.random.default_rng(np.random.SeedSequence([seed, scenario_index])).choice(starts))
    offsets = np.arange(config['window_size'])
    samples = config.get('mc_dropout_samples', config.get('ucb_mc_samples', config.get('vcl_evaluation_mc_samples')))
    if model_name != 'ensemble' and samples is None:
        raise ValueError(f'{model_name} does not expose sampled epistemic predictions for cross-simulation testing.')
    windows = torch.from_numpy(np.asarray(dataset['data'][selected_start + offsets])).float().unsqueeze(0).to(device)
    predicted, epistemic = predict_with_epistemic(model_name, evaluator, model, windows, samples)
    predicted, epistemic = predicted.cpu().numpy(), epistemic.cpu().numpy()
    final_index = selected_start + config['window_size'] - 1
    contacts = np.asarray(dataset['labels'][final_index])
    ground_truth = np.asarray(dataset['velocities'][final_index])
    if contacts.ndim == 0:
        contacts = contacts.reshape(1, 1)
    else:
        contacts = contacts[None, :]
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[None]
    contact_mask = contacts == 1
    mean_epistemic = float(epistemic[contact_mask].mean())
    return {
        'seed': seed, 'simulator': dataset['name'],
        'cmd_vel_x': float(dataset['data'][final_index, -1]),
        'velocity_mae': float(np.abs(predicted - ground_truth)[contact_mask].mean()),
        'mean_epistemic_variance': mean_epistemic,
        'log10_mean_epistemic_variance': float(np.log10(max(mean_epistemic, np.finfo(float).tiny))),
    }


def save_plot(rows, output_path, scenario_order=TEST_SCENARIOS, by_environment=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, (epistemic_axis, mae_axis) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for scenario_index, scenario in enumerate(scenario_order):
        source_rows = [row for row in rows if row['simulator'] == scenario]
        command_velocity = ([scenario_index] * len(source_rows) if by_environment
                            else [float(row['cmd_vel_x']) for row in source_rows])
        log10_epistemic = [float(row['log10_mean_epistemic_variance']) for row in source_rows]
        mae = [float(row['velocity_mae']) for row in source_rows]
        epistemic_axis.scatter(command_velocity, log10_epistemic, s=30,
                               label=f'{scenario} (mean log10 epi={np.mean(log10_epistemic):.2f})', **SCENARIO_STYLES[scenario])
        mae_axis.scatter(command_velocity, mae, s=30,
                         label=f'{scenario} (mean MAE={np.mean(mae):.2e})', **SCENARIO_STYLES[scenario])
    epistemic_axis.set(ylabel='log10(mean epistemic variance)')
    mae_axis.set(xlabel='MuJoCo environment' if by_environment else 'Command velocity x (m/s)', ylabel='Contact-masked velocity MAE')
    if by_environment:
        mae_axis.set_xticks(range(len(scenario_order)), scenario_order)
    for axis in (epistemic_axis, mae_axis):
        axis.grid(True, which='both', alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_cross_sim_evaluation(model_name, config_path, scenarios, checkpoint_path, first_seed, last_seed,
                             output_csv, save_csv=True, save_png=True, by_environment=False):
    """Shared cross-simulation evaluation used by standalone and replay training."""
    if first_seed > last_seed:
        raise ValueError('Use an ordered seed range.')
    unknown_scenarios = set(scenarios) - set(SCENARIO_FOLDERS)
    if unknown_scenarios:
        raise ValueError(f'Unknown scenario(s): {sorted(unknown_scenarios)}')
    missing_styles = set(scenarios) - set(SCENARIO_STYLES)
    if missing_styles:
        raise ValueError(f'Missing plot styles for scenario(s): {sorted(missing_styles)}')
    datasets = [load_dataset(name, SCENARIO_FOLDERS[name]) for name in scenarios]
    reference = datasets[0]
    for dataset in datasets[1:]:
        if (dataset['data'].shape[1] != reference['data'].shape[1]
                or dataset['metadata']['legs'] != reference['metadata']['legs']):
            raise ValueError(f"{dataset['name']} must match {reference['name']} feature dimensions and leg order.")
    evaluator, config, device, model, checkpoint = load_evaluator_and_model(
        model_name, config_path, reference['data'].shape[1], reference['metadata']['legs'], checkpoint_path
    )
    print(f'Loaded {model_name} checkpoint(s): {checkpoint}')
    rows = []
    for seed in range(first_seed, last_seed + 1):
        rows.extend(evaluate_dataset(model_name, evaluator, config, device, model, dataset, seed, scenario_index)
                    for scenario_index, dataset in enumerate(datasets))
    output_csv = Path(output_csv)
    if save_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open('w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        print(f'Wrote {len(rows)} rows to {output_csv}')
    if save_png:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_plot(rows, output_csv.with_suffix('.png'), scenarios, by_environment)
        print(f'Wrote {output_csv.with_suffix(".png")}')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-seed', type=int, default=FIRST_SEED)
    parser.add_argument('--last-seed', type=int, default=LAST_SEED)
    parser.add_argument('--checkpoint-path', type=Path, default=CHECKPOINT_PATH)
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--no-save-csv', action='store_true')
    parser.add_argument('--no-save-plot', action='store_true')
    args = parser.parse_args(argv)
    if args.first_seed > args.last_seed:
        raise ValueError('Use an ordered seed range.')

    output_dir = _ROOT / 'testResults' / 'CrossSim'
    scenario_suffix = '_'.join(TEST_SCENARIOS).lower()
    output_csv = args.output_csv or output_dir / f'{MODEL}_{scenario_suffix}_seeds_{args.first_seed}-{args.last_seed}.csv'
    run_cross_sim_evaluation(MODEL, _ROOT / 'config' / 'network_params.yaml', TEST_SCENARIOS,
                             args.checkpoint_path, args.first_seed, args.last_seed, output_csv,
                             SAVE_CSV and not args.no_save_csv, SAVE_PNG and not args.no_save_plot)


if __name__ == '__main__':
    main()
