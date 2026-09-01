"""Build the run-safe task index used by train/trainReplay.py."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from utils.ucb_task_windows import _validate_task_windows, build_task_window_index


def main():
    root = _ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-config", type=Path, default=root / "config/Replay_params.yaml")
    parser.add_argument("--data-folder", type=Path, default=root / "Data/NumpyFiles")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    args = parser.parse_args()
    with args.replay_config.open() as file:
        config = yaml.safe_load(file) or {}
    if config.get("continual_feature") != "cmd_vel_x":
        raise ValueError("Only continual_feature: cmd_vel_x is supported.")
    if args.window_size is None:
        with (root / "config/network_params.yaml").open() as file:
            window_size = (yaml.safe_load(file) or {})["window_size"]
    else:
        window_size = args.window_size
    data = np.load(args.data_folder / "all_data.npy")
    boundaries = np.load(args.data_folder / "all_data_boundaries.npy")
    windows = _validate_task_windows(config["continual_task_windows"])
    starts, task_ids, cmd_velocities, run_ids = build_task_window_index(data, boundaries, window_size, windows)
    output = args.output or args.data_folder / "replay_task_windows.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, task_windows=windows, window_size=np.int64(window_size), window_starts=starts,
                        task_ids=task_ids, window_cmd_vel_x=cmd_velocities, window_run_ids=run_ids)
    print(f"Saved {len(starts)} replay task windows to {output}")


if __name__ == "__main__":
    main()
