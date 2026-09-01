"""Build ordered cmd_vel_x task-window indices from the canonical NumPy dataset."""

import argparse
from pathlib import Path

import numpy as np
import yaml


def _validate_task_windows(task_windows):
    windows = np.asarray(task_windows, dtype=np.float64)
    if windows.ndim != 2 or windows.shape[1] != 2 or np.any(windows[:, 0] >= windows[:, 1]):
        raise ValueError("continual_task_windows must contain [low, high] pairs with low < high.")
    if np.any(windows[1:, 0] < windows[:-1, 1]):
        raise ValueError("continual_task_windows must be ordered and non-overlapping.")
    return windows


def build_task_window_index(data, boundaries, window_size, task_windows, cmd_vel_column=-1):
    """Return valid sliding-window starts, task IDs, and final-timestep cmd_vel_x values.

    Task windows are always left-closed/right-open: ``low <= cmd_vel_x < high``.
    """
    data = np.asarray(data)
    boundaries = np.asarray(boundaries, dtype=np.int64)
    windows = _validate_task_windows(task_windows)
    if data.ndim != 2 or not 0 < window_size or not len(boundaries):
        raise ValueError("Expected 2D data, a positive window_size, and non-empty run boundaries.")
    if np.any(boundaries <= 0) or np.any(boundaries[1:] <= boundaries[:-1]) or boundaries[-1] != len(data):
        raise ValueError("Run boundaries must be strictly increasing and end at len(data).")

    starts, task_ids, cmd_velocities, run_ids = [], [], [], []
    run_starts = np.r_[0, boundaries[:-1]]
    for run_id, (run_start, run_end) in enumerate(zip(run_starts, boundaries)):
        window_starts = np.arange(run_start, run_end - window_size + 1, dtype=np.int64)
        if not len(window_starts):
            continue
        values = data[window_starts + window_size - 1, cmd_vel_column]
        matches = (values[:, None] >= windows[:, 0]) & (values[:, None] < windows[:, 1])
        if np.any(matches.sum(axis=1) > 1):
            raise ValueError("A task window matched more than once.")
        selected, selected_tasks = np.nonzero(matches)
        starts.append(window_starts[selected])
        task_ids.append(selected_tasks.astype(np.int64))
        cmd_velocities.append(values[selected])
        run_ids.append(np.full(len(selected), run_id, dtype=np.int64))

    if not starts:
        return tuple(np.empty(0, dtype=np.int64) for _ in range(4))
    return tuple(np.concatenate(values) for values in (starts, task_ids, cmd_velocities, run_ids))


def task_window_starts(index, task_id):
    """Return raw all_data.npy start indices for one ordered UCB task."""
    return index["window_starts"][index["task_ids"] == task_id]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create cmd_vel_x UCB task-window indices.")
    parser.add_argument("--ucb-config", type=Path, default=root / "config/UCB_params.yaml")
    parser.add_argument("--data-folder", type=Path, default=root / "Data/NumpyFiles")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    args = parser.parse_args()

    with args.ucb_config.open() as config_file:
        config = yaml.safe_load(config_file) or {}
    if config.get("continual_feature") != "cmd_vel_x":
        raise ValueError("Only continual_feature: cmd_vel_x is supported.")
    window_size = args.window_size
    if window_size is None:
        with (root / "config/network_params.yaml").open() as config_file:
            window_size = (yaml.safe_load(config_file) or {})["window_size"]

    data = np.load(args.data_folder / "all_data.npy")
    boundaries = np.load(args.data_folder / "all_data_boundaries.npy")
    task_windows = _validate_task_windows(config["continual_task_windows"])
    window_starts, task_ids, cmd_velocities, run_ids = build_task_window_index(
        data, boundaries, window_size, task_windows
    )
    output = args.output or args.data_folder / "ucb_task_windows.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        task_windows=task_windows,
        window_size=np.int64(window_size),
        window_starts=window_starts,
        task_ids=task_ids,
        window_cmd_vel_x=cmd_velocities,
        window_run_ids=run_ids,
    )
    print(f"Saved {len(window_starts)} task windows to {output}")
    for task_id, (low, high) in enumerate(task_windows):
        print(f"Task {task_id} [{low:g}, {high:g}): {(task_ids == task_id).sum()} windows")


if __name__ == "__main__":
    main()
