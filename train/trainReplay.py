"""Train a deterministic Gaussian TCN task-by-task with balanced replay."""

import argparse
import csv
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from contact_cnn import ContactCNNWithNormalization, EnsembleTCN
from train.base_train import BaseTrainer
from utils.ucb_task_windows import _validate_task_windows


class WindowDataset(Dataset):
    def __init__(self, data, labels, velocities, starts, window_size):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.velocities = torch.as_tensor(velocities, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window_size = int(window_size)
        if not len(self.starts):
            raise ValueError("A replay task split has no windows.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start, end = int(self.starts[index]), int(self.starts[index]) + self.window_size
        return {"data": self.data[start:end], "label": self.labels[end - 1],
                "label_seq": self.labels[start:end], "velocity": self.velocities[start:end]}


def _load_config(network_path, replay_path):
    with network_path.open() as file:
        config = yaml.safe_load(file) or {}
    with replay_path.open() as file:
        config.update(yaml.safe_load(file) or {})
    data_folder = Path(config["data_folder"])
    config["data_folder"] = str(data_folder if data_folder.is_absolute() else _ROOT / data_folder)
    return config


def _load_index(path, task_windows, window_size):
    with np.load(path) as index:
        required = {"task_windows", "window_starts", "task_ids", "window_run_ids"}
        if missing := required.difference(index.files):
            raise ValueError(f"Task index is missing {sorted(missing)}; rebuild it with utils/ucb_task_windows.py.")
        if not np.array_equal(index["task_windows"], task_windows):
            raise ValueError("Task index windows differ from config/Replay_params.yaml; rebuild the index.")
        if "window_size" in index and int(index["window_size"]) != int(window_size):
            raise ValueError("Task index window size differs from network_params.yaml; rebuild the index.")
        return {name: index[name].copy() for name in required}


def _split_task(index, task_id, train_ratio, seed):
    mask = index["task_ids"] == task_id
    starts, run_ids = index["window_starts"][mask], index["window_run_ids"][mask]
    runs = np.unique(run_ids)
    if len(runs) < 2:
        raise ValueError(f"Task {task_id} needs windows from at least two runs for a run-disjoint validation split.")
    rng = np.random.default_rng(seed + task_id)
    rng.shuffle(runs)
    train_count = min(len(runs) - 1, max(1, math.floor(train_ratio * len(runs))))
    train = starts[np.isin(run_ids, runs[:train_count])]
    validation = starts[np.isin(run_ids, runs[train_count:])]
    if not len(train) or not len(validation):
        raise ValueError(f"Task {task_id} split produced {len(train)} train and {len(validation)} validation windows.")
    return train, validation


def _normalization_stats(data, starts, window_size, device):
    windows = torch.as_tensor(data[starts[:, None] + np.arange(window_size)], dtype=torch.float32)
    flattened = windows.reshape(-1, windows.shape[-1])
    low, high = torch.quantile(flattened, .01, dim=0), torch.quantile(flattened, .99, dim=0)
    clipped = flattened.clamp(low, high)
    return (clipped.mean(0).view(1, 1, -1).to(device),
            clipped.std(0).clamp_min(1e-8).view(1, 1, -1).to(device))


def _loader(data, labels, velocities, starts, config, shuffle, seed):
    return DataLoader(WindowDataset(data, labels, velocities, starts, config["window_size"]),
                      batch_size=config["batch_size"], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


class ReplayTrainer(BaseTrainer):
    """BaseTrainer's ordinary Gaussian workflow, reset once per continual task."""
    logs_dir = _ROOT / "logs" / "logsReplay"

    def velocity_loss(self, outputs, velocity, dense):
        prediction, covariance = outputs[0], outputs[2]
        if dense:
            prediction, covariance = prediction.permute(0, 3, 1, 2), covariance.permute(0, 3, 1, 2)
        else:
            prediction, covariance = outputs[1], outputs[3]
        return torch.nn.functional.gaussian_nll_loss(prediction, velocity, covariance, reduction="none")


def _balanced_replay(task_train_starts, capacity, seed):
    """Sample a fixed, evenly task-balanced memory from completed task training sets."""
    if capacity < 1:
        return np.empty(0, dtype=np.int64)
    rng, count = np.random.default_rng(seed), len(task_train_starts)
    quotas = np.full(count, capacity // count, dtype=int)
    quotas[:capacity % count] += 1
    return np.concatenate([starts[rng.choice(len(starts), min(len(starts), quota), replace=False)]
                           for starts, quota in zip(task_train_starts, quotas)])


def _evaluate_task_checkpoint(run_dir, task_id, checkpoint, config, model_name, results_name, plot_metric):
    results_dir = _ROOT / "testResults" / results_name / run_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / f"{model_name}_series_after_task_{task_id:02d}.csv"
    subprocess.run([
        sys.executable, _ROOT / "tests/base_testSeries.py", "--model", model_name,
        "--checkpoint-path", checkpoint, "--cmd-vel-x-window",
        str(config["evaluation_window"][0]), str(config["evaluation_window"][1]),
        "--first-seed", str(config["replay_evaluation_first_seed"]),
        "--last-seed", str(config["replay_evaluation_last_seed"]),
        "--output-csv", output_csv, "--save-csv", "--plot-metric", plot_metric, "--skip-knn", "--quiet",
        "--fail-if-output-exists",
    ], cwd=_ROOT, check=True)


def run_replay_training(config, task_index, model_factory, trainer_type, logs_name, test_model,
                        test_results_name, plot_metric, dry_run=False, skip_evaluate_after_task=False):
    """Shared ordered replay lifecycle; model-specific code is supplied by callers."""
    tasks = _validate_task_windows(config["continual_task_windows"])
    index = _load_index(task_index, tasks, config["window_size"])
    data_folder = Path(config["data_folder"])
    data, labels, velocities = (np.load(data_folder / name) for name in ("all_data.npy", "all_labels.npy", "all_body_velocities.npy"))
    seed = int(config.get("random_seed", 42))
    splits = [_split_task(index, task_id, float(config.get("train_ratio", .85)), seed) for task_id in range(len(tasks))]
    for task_id, (train, validation) in enumerate(splits):
        print(f"Task {task_id} [{tasks[task_id, 0]:g}, {tasks[task_id, 1]:g}): {len(train)} train, {len(validation)} validation")
    if dry_run:
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = _normalization_stats(data, splits[0][0], config["window_size"], device)
    model = model_factory(config, data.shape[1], mean, std).to(device)
    run_dir = _ROOT / "logs" / logs_name / f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    run_dir.mkdir(parents=True)
    with (run_dir / "network_params.yaml").open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    rows, completed_train, replay = [], [], np.empty(0, dtype=np.int64)
    for task_id, (task_train, task_validation) in enumerate(splits):
        training_starts = np.concatenate((task_train, replay))
        train_loader = _loader(data, labels, velocities, training_starts, config, True, seed + task_id)
        validation_loader = _loader(data, labels, velocities, task_validation, config, False, seed)
        print(f"Task {task_id}: {len(task_train)} current + {len(replay)} replay windows")
        task_dir = run_dir / f"task_{task_id:02d}"
        trainer = trainer_type(model, config.copy(), str(task_dir))
        trainer.train(train_loader, validation_loader)
        best_checkpoint = task_dir / "model_best_val_loss.pt"
        best_state = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])
        completed_train.append(task_train)
        replay = _balanced_replay(completed_train, int(config["replay_capacity"]), int(config.get("replay_seed", seed)))
        checkpoint = run_dir / f"model_after_task_{task_id:02d}.pt"
        torch.save({"model_state_dict": model.state_dict(), "task_id": task_id, "next_task_id": task_id + 1,
                    "best_validation_loss": best_state["val_loss"], "best_epoch": best_state["epoch"],
                    "replay_window_starts": replay.tolist(), "task_train_window_starts": [x.tolist() for x in completed_train],
                    "config": config}, checkpoint)
        if config.get("replay_evaluate_after_task", True) and not skip_evaluate_after_task:
            _evaluate_task_checkpoint(run_dir, task_id, checkpoint, config, test_model, test_results_name, plot_metric)
        for eval_task, (_, eval_starts) in enumerate(splits[:task_id + 1]):
            metrics = trainer.evaluate(_loader(data, labels, velocities, eval_starts, config, False, seed))
            metrics["loss"] = metrics["contact_loss"] + metrics["velocity_loss"]
            rows.append({"checkpoint_after_task": task_id, "evaluation_task": eval_task, **metrics})
        with (run_dir / "task_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        print(f"Saved {checkpoint} and {run_dir / 'task_metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=Path, default=_ROOT / "config/network_params.yaml")
    parser.add_argument("--replay-config", type=Path, default=_ROOT / "config/Replay_params.yaml")
    parser.add_argument("--task-index", type=Path, default=_ROOT / "Data/NumpyFiles/replay_task_windows.npz")
    parser.add_argument("--dry-run", action="store_true", help="Print task splits and exit.")
    parser.add_argument("--skip-evaluate-after-task", action="store_true", help="Do not run testReplay after each task checkpoint.")
    args = parser.parse_args()
    config = _load_config(args.config_name, args.replay_config)
    if config.get("model_architecture", "tcn").lower() != "tcn":
        raise ValueError("Replay training requires model_architecture: 'tcn'.")
    run_replay_training(
        config, args.task_index,
        lambda cfg, features, mean, std: ContactCNNWithNormalization(EnsembleTCN(
            cfg["window_size"], features, cfg.get("tcn_num_channels", 64), cfg.get("tcn_kernel_size", 3),
            cfg.get("tcn_num_blocks", 5), cfg.get("tcn_dropout", .2)), mean, std),
        ReplayTrainer, "logsReplay", "replay", "Replay", "velocity-mae",
        args.dry_run, args.skip_evaluate_after_task,
    )


if __name__ == "__main__":
    main()
