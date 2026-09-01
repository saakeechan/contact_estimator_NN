"""Train the shared-head UCB Bayes-by-Backprop velocity model task by task."""

import argparse
import copy
import math
import os
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from contact_cnn import ContactCNNWithNormalization, UCBTCN
from utils.ucb_task_windows import _validate_task_windows


class TaskWindowDataset(Dataset):
    """Raw, run-safe windows selected by the precomputed UCB task index."""
    def __init__(self, data, labels, velocities, starts, window_size):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.velocities = torch.as_tensor(velocities, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window_size = int(window_size)
        if not len(self.starts):
            raise ValueError("A UCB task split has no windows.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        end = start + self.window_size
        return {
            "data": self.data[start:end],
            "label": self.labels[end - 1],
            "label_seq": self.labels[start:end],
            "velocity": self.velocities[start:end],
        }


class UncertaintyScaledSGD:
    """SGD with Algorithm 2's persistent, element-wise mean learning rates."""
    def __init__(self, model, learning_rate, mean_lr_multipliers=None):
        self.model = model
        self.learning_rate = float(learning_rate)
        self.mean_lr_multipliers = {
            name: multiplier.to(device=parameter.device, dtype=parameter.dtype)
            for name, parameter in model.named_parameters()
            for multiplier in [(mean_lr_multipliers or {}).get(name, torch.ones_like(parameter))]
            if name.endswith("_mu")
        }

    @torch.no_grad()
    def step(self):
        for module_name, module in self.model.named_modules():
            if not hasattr(module, "weight_mu") or not hasattr(module, "weight_rho"):
                continue
            for parameter_name, mu, rho in (
                (f"{module_name}.weight_mu", module.weight_mu, module.weight_rho),
                (f"{module_name}.bias_mu", module.bias_mu, module.bias_rho),
            ):
                if mu is None:
                    continue
                if mu.grad is not None:
                    mu.addcmul_(mu.grad, self.mean_lr_multipliers[parameter_name], value=-self.learning_rate)
                if rho.grad is not None:
                    rho.add_(rho.grad, alpha=-self.learning_rate)
                    # Keep the variational scale in the finite range used by the model.
                    rho.nan_to_num_(nan=-3.0, posinf=5.0, neginf=-12.0).clamp_(-12.0, 5.0)

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def state_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "mean_lr_multipliers": {name: value.detach().cpu() for name, value in self.mean_lr_multipliers.items()},
        }

    def load_state_dict(self, state):
        self.learning_rate = float(state["learning_rate"])
        for name, value in state.get("mean_lr_multipliers", {}).items():
            if name in self.mean_lr_multipliers:
                self.mean_lr_multipliers[name] = value.to(self.mean_lr_multipliers[name])


@torch.no_grad()
def _update_ucb_mean_lr_multipliers(model, multipliers):
    """Apply Algorithm 2 after a completed task: alpha_mu <- alpha_mu * sigma."""
    updated = {}
    for module_name, module in model.named_modules():
        if not hasattr(module, "weight_mu") or not hasattr(module, "weight_rho"):
            continue
        for parameter_name, mu, rho in (
            (f"{module_name}.weight_mu", module.weight_mu, module.weight_rho),
            (f"{module_name}.bias_mu", module.bias_mu, module.bias_rho),
        ):
            if mu is None:
                continue
            # Match the finite posterior scale used by contact_cnn.py.
            sigma = F.softplus(torch.nan_to_num(rho, nan=-3.0, posinf=5.0, neginf=-12.0).clamp(-12.0, 5.0)).clamp_min(1e-6)
            updated[parameter_name] = multipliers.get(parameter_name, torch.ones_like(mu)).to(mu) * sigma
    return updated


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _load_config(root, network_config, ucb_config):
    with network_config.open() as file:
        config = yaml.safe_load(file) or {}
    with ucb_config.open() as file:
        config.update(yaml.safe_load(file) or {})
    config["data_folder"] = str(_resolve(root, config["data_folder"]))
    return config


def _load_vcl_checkpoint(model, state_dict, next_task_id):
    """Load a VCL checkpoint, allowing only a task-0 legacy checkpoint upgrade."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        name for name in model.state_dict()
        if name.endswith(("weight_prior_mu", "weight_prior_sigma", "bias_prior_mu", "bias_prior_sigma", "uses_previous_posterior_prior"))
    }
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - allowed_missing:
        raise RuntimeError(
            f"Checkpoint is incompatible with the UCB model; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    if incompatible.missing_keys:
        if next_task_id != 1:
            raise ValueError(
                "This pre-VCL checkpoint has no previous-task prior snapshot and cannot safely resume beyond task 1."
            )
        # A legacy task-0 checkpoint can be upgraded unambiguously: its current q_1 becomes p_2.
        model.base_model.snapshot_posterior_as_prior()


def _load_task_index(index_path, task_windows, window_size):
    with np.load(index_path) as index:
        required = {"task_windows", "window_starts", "task_ids", "window_run_ids"}
        missing = required.difference(index.files)
        if missing:
            raise ValueError(f"Task index is missing: {sorted(missing)}. Rebuild it with utils/ucb_task_windows.py.")
        indexed_windows = index["task_windows"]
        if not np.array_equal(indexed_windows, task_windows):
            if len(task_windows) > len(indexed_windows) or not np.array_equal(indexed_windows[:len(task_windows)], task_windows):
                raise ValueError("Task index windows do not match config/UCB_params.yaml; rebuild the index.")
        if "window_size" in index and int(index["window_size"]) != int(window_size):
            raise ValueError("Task index window size does not match the active network config; rebuild the index.")
        values = {name: index[name].copy() for name in required}
        # A configuration may train only an initial prefix of an existing task index.
        selected = values["task_ids"] < len(task_windows)
        return {name: value[selected] if name != "task_windows" else value for name, value in values.items()}


def _split_task_windows(index, task_id, train_ratio, data_fraction, seed):
    mask = index["task_ids"] == task_id
    starts, run_ids = index["window_starts"][mask], index["window_run_ids"][mask]
    if not 0 < data_fraction <= 1:
        raise ValueError("ucb_data_fraction must be in (0, 1].")
    count = max(1, int(math.floor(len(starts) * data_fraction)))
    if count < len(starts):
        selected = np.random.default_rng(seed + task_id).choice(len(starts), size=count, replace=False)
        starts, run_ids = starts[np.sort(selected)], run_ids[np.sort(selected)]
    unique_runs = np.unique(run_ids)
    if len(unique_runs) < 2:
        raise ValueError(f"Task {task_id} needs windows from at least two runs for a run-disjoint validation split.")
    generator = np.random.default_rng(seed + task_id)
    generator.shuffle(unique_runs)
    train_count = min(len(unique_runs) - 1, max(1, int(math.floor(train_ratio * len(unique_runs)))))
    train_runs = set(unique_runs[:train_count])
    train_starts, val_starts = starts[np.isin(run_ids, list(train_runs))], starts[~np.isin(run_ids, list(train_runs))]
    if not len(train_starts) or not len(val_starts):
        raise ValueError(f"Task {task_id} split produced {len(train_starts)} train and {len(val_starts)} validation windows.")
    return train_starts, val_starts


def _normalization_stats(data, starts, window_size, device):
    offsets = np.arange(window_size)
    windows = torch.as_tensor(data[starts[:, None] + offsets], dtype=torch.float32)
    flattened = windows.reshape(-1, windows.shape[-1])
    low, high = torch.quantile(flattened, .01, dim=0), torch.quantile(flattened, .99, dim=0)
    clipped = flattened.clamp(low, high)
    return clipped.mean(dim=0).view(1, 1, -1).to(device), clipped.std(dim=0).clamp_min(1e-8).view(1, 1, -1).to(device)


def _make_loaders(data, labels, velocities, train_starts, val_starts, config, seed):
    train = TaskWindowDataset(data, labels, velocities, train_starts, config["window_size"])
    val = TaskWindowDataset(data, labels, velocities, val_starts, config["window_size"])
    return (
        DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed)),
        DataLoader(val, batch_size=config.get("test_batch_size", config["batch_size"]), shuffle=False),
    )


def _likelihood_nll(outputs, sample, dense, contact_weight, velocity_weight, reduction="mean"):
    velocity_target = sample["velocity"] if dense else sample["velocity"][:, -1]
    contact_target = sample["label_seq"] if dense else sample["label"]
    if dense:
        velocity_prediction, variance = outputs[0].permute(0, 3, 1, 2), outputs[2].permute(0, 3, 1, 2)
    else:
        velocity_prediction, variance = outputs[1], outputs[3]
    velocity_nll = F.gaussian_nll_loss(velocity_prediction, velocity_target, variance, reduction="none")
    contact_mask = (contact_target == 1).float().unsqueeze(-1)
    velocity_nll = (velocity_nll * contact_mask).sum()
    contact_nll = F.binary_cross_entropy_with_logits(outputs[4], sample["label"], reduction="sum")
    if reduction == "mean":
        velocity_nll = velocity_nll / (contact_mask.sum() * velocity_target.shape[-1] + 1e-8)
        contact_nll = contact_nll / sample["label"].numel()
    elif reduction != "sum":
        raise ValueError("reduction must be 'mean' or 'sum'.")
    return contact_weight * contact_nll + velocity_weight * velocity_nll, contact_nll, velocity_nll


@torch.no_grad()
def _evaluate(model, loader, device, config):
    model.eval()
    total = contact = velocity = absolute_error = contact_components = 0.0
    dense = config.get("use_dense_supervision", False)
    for sample in loader:
        sample = {name: value.to(device) for name, value in sample.items()}
        outputs = model(sample["data"], return_sequence=dense)
        loss, contact_loss, velocity_loss = _likelihood_nll(
            outputs, sample, dense,
            float(config.get("contact_weight", 1.0)), float(config.get("velocity_weight", 1.0)),
        )
        prediction, target = (outputs[0].permute(0, 3, 1, 2), sample["velocity"]) if dense else (outputs[1], sample["velocity"][:, -1])
        mask = ((sample["label_seq"] if dense else sample["label"]) == 1).float().unsqueeze(-1)
        absolute_error += ((prediction - target).abs() * mask).sum().item()
        contact_components += mask.sum().item() * target.shape[-1]
        total += loss.item(); contact += contact_loss.item(); velocity += velocity_loss.item()
    batches = max(1, len(loader))
    return {"loss": total / batches, "contact_loss": contact / batches, "velocity_loss": velocity / batches,
            "velocity_mae": absolute_error / max(contact_components, 1.0)}


def _train_task(model, task_id, train_loader, val_loader, config, device, mean_lr_multipliers):
    """Optimize the minibatch BBB objective for one ordered continual task."""
    learning_rate = float(config.get("ucb_init_lr", config["init_lr"]))
    optimizer = UncertaintyScaledSGD(model, learning_rate, mean_lr_multipliers)
    samples = int(config["ucb_mc_samples"])
    if samples < 1:
        raise ValueError("ucb_mc_samples must be positive.")
    minibatches = len(train_loader)
    if minibatches < 1:
        raise ValueError(f"Task {task_id} has no minibatches.")
    patience_limit = int(config.get("ucb_lr_patience", 5))
    patience, lr_updates, best = patience_limit, 0, {"loss": float("inf"), "state": None, "epoch": -1}
    history = []
    for epoch in range(int(config["num_epoch"])):
        model.train()
        total = data_nll = kl = 0.0
        progress = tqdm(train_loader, desc=f"Task {task_id} epoch {epoch + 1}/{config['num_epoch']}", unit="batch")
        for sample in progress:
            sample = {name: value.to(device) for name, value in sample.items()}
            losses, log_priors, log_posts = [], [], []
            for _ in range(samples):
                outputs = model(sample["data"], return_sequence=config.get("use_dense_supervision", False), sample=True, calculate_log_probs=True)
                likelihood, _, _ = _likelihood_nll(
                    outputs, sample, config.get("use_dense_supervision", False),
                    float(config.get("contact_weight", 1.0)), float(config.get("velocity_weight", 1.0)),
                    reduction="sum",
                )
                log_prior, log_posterior = model.base_model.bayesian_log_probs()
                losses.append(likelihood); log_priors.append(log_prior); log_posts.append(log_posterior)
            mean_nll, mean_kl = torch.stack(losses).mean(), torch.stack(log_posts).mean() - torch.stack(log_priors).mean()
            # L_BBB = E_q[-log p(D_batch|w)] + E_q[log q(w)-log p(w)] / M.
            loss = mean_nll + mean_kl / minibatches
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("ucb_gradient_clip_norm", 10.0)))
            optimizer.step()
            total += loss.item(); data_nll += mean_nll.item(); kl += mean_kl.item()
            progress.set_postfix(bbb=f"{loss.item():.3g}", nll=f"{mean_nll.item():.3g}", kl=f"{mean_kl.item() / minibatches:.3g}")
        train_evaluation = _evaluate(model, train_loader, device, config)
        validation = _evaluate(model, val_loader, device, config)
        train_metrics = {"loss": total / minibatches, "data_nll": data_nll / minibatches, "kl": kl / minibatches}
        history.append({"epoch": epoch, "learning_rate": optimizer.learning_rate, **train_metrics,
                        **{f"train_{key}": value for key, value in train_evaluation.items()},
                        **{f"val_{key}": value for key, value in validation.items()}})
        print(f"Task {task_id} epoch {epoch + 1}/{config['num_epoch']} | train BBB {train_metrics['loss']:.5f} | train MAE {train_evaluation['velocity_mae']:.5f} | val NLL {validation['loss']:.5f} | val MAE {validation['velocity_mae']:.5f} | lr {optimizer.learning_rate:.3e}")
        if validation["loss"] < best["loss"]:
            best = {"loss": validation["loss"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            patience = patience_limit
        else:
            patience -= 1
            if patience <= 0:
                learning_rate /= float(config.get("ucb_lr_factor", 3.0))
                if learning_rate < float(config.get("ucb_min_lr", 1e-6)) or lr_updates >= int(config.get("ucb_max_lr_updates", 3)):
                    break
                optimizer.learning_rate = learning_rate
                lr_updates += 1
                patience = patience_limit
    model.load_state_dict(best["state"])
    # VCL transition: retain this detached posterior as the next task's prior.
    model.base_model.snapshot_posterior_as_prior()
    next_mean_lr_multipliers = _update_ucb_mean_lr_multipliers(model, optimizer.mean_lr_multipliers)
    optimizer.mean_lr_multipliers = next_mean_lr_multipliers
    return best, history, optimizer.state_dict()


def _checkpoint(path, model, task_id, train_starts, config, optimizer_state, best, history):
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer_state,
        "task_id": task_id, "next_task_id": task_id + 1, "train_window_starts": np.asarray(train_starts).tolist(),
        "ucb_config": config, "best_validation_loss": best["loss"], "best_epoch": best["epoch"], "history": history,
    }, path)


def _evaluate_task_checkpoint(root, run_dir, task_id, checkpoint_path, config):
    results_dir = root / "testResults" / "UCB" / run_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / f"ucb_series_after_task_{task_id:02d}.csv"
    command = [
        sys.executable, root / "tests/base_testSeries.py", "--model", "ucb",
        "--checkpoint-path", checkpoint_path, "--cmd-vel-x-window",
        str(config["evaluation_window"][0]), str(config["evaluation_window"][1]),
        "--first-seed", str(config.get("ucb_evaluation_first_seed", 203)),
        "--last-seed", str(config.get("ucb_evaluation_last_seed", 203)),
        "--output-csv", output_csv, "--skip-knn", "--quiet",
        "--fail-if-output-exists",
    ]
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=Path, default=_ROOT / "config/network_params.yaml")
    parser.add_argument("--ucb-config", type=Path, default=_ROOT / "config/UCB_params.yaml")
    parser.add_argument("--task-index", type=Path, default=_ROOT / "Data/NumpyFiles/ucb_task_windows.npz")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume from a completed-task UCB checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print task splits without training.")
    parser.add_argument("--evaluate-after-task", action="store_true", help="Run the configured UCB series evaluation after each checkpoint.")
    args = parser.parse_args()
    config = _load_config(_ROOT, args.config_name, args.ucb_config)
    task_windows = _validate_task_windows(config["continual_task_windows"])
    index = _load_task_index(args.task_index, task_windows, config["window_size"])
    data_folder = Path(config["data_folder"])
    data = np.load(data_folder / "all_data.npy")
    labels = np.load(data_folder / "all_labels.npy")
    velocities = np.load(data_folder / "all_body_velocities.npy")
    if labels.shape != (len(data), 1) or velocities.shape != (len(data), 1, 3):
        raise ValueError("Expected all_labels.npy (N,1) and all_body_velocities.npy (N,1,3).")
    starts = index["window_starts"]
    if np.any(starts < 0) or np.any(starts + config["window_size"] > len(data)):
        raise ValueError("Task index contains starts outside all_data.npy; rebuild it with utils/ucb_task_windows.py.")
    seed = int(config.get("random_seed", 42))
    data_fraction = float(config.get("ucb_data_fraction", 1.0))
    splits = []
    for task_id in range(len(task_windows)):
        splits.append(_split_task_windows(
            index, task_id, float(config.get("train_ratio", .85)), data_fraction, seed
        ))
    for task_id, (train_starts, val_starts) in enumerate(splits):
        print(f"Task {task_id} [{task_windows[task_id, 0]:g}, {task_windows[task_id, 1]:g}): {len(train_starts)} train, {len(val_starts)} validation windows")
    if args.dry_run:
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_task, checkpoint = 0, None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        start_task = int(checkpoint["next_task_id"])
        run_dir = args.resume_checkpoint.resolve().parent
    else:
        run_dir = _ROOT / "logsUCB" / f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        run_dir.mkdir(parents=True, exist_ok=False)
        with (run_dir / "network_params.yaml").open("w") as file:
            yaml.safe_dump(config, file, sort_keys=False)
    if start_task >= len(task_windows):
        raise ValueError("The resume checkpoint already completed every configured task.")
    stats_starts = splits[0][0]
    mean, std = _normalization_stats(data, stats_starts, config["window_size"], device)
    model = ContactCNNWithNormalization(UCBTCN(
        config["window_size"], data.shape[1], config.get("tcn_num_channels", 64), config.get("tcn_kernel_size", 3),
        config.get("tcn_num_blocks", 5), config.get("tcn_dropout", .2), config.get("ucb_rho", -3.0),
        config.get("ucb_sig1", 0.0), config.get("ucb_sig2", 6.0), config.get("ucb_pi", .25),
    ), global_mean=mean, global_std=std).to(device)
    mean_lr_multipliers = {}
    if checkpoint is not None:
        _load_vcl_checkpoint(model, checkpoint["model_state_dict"], start_task)
        mean_lr_multipliers = checkpoint.get("optimizer_state_dict", {}).get("mean_lr_multipliers", {})
    for task_id in range(start_task, len(task_windows)):
        train_starts, val_starts = splits[task_id]
        train_loader, val_loader = _make_loaders(data, labels, velocities, train_starts, val_starts, config, seed + task_id)
        best, history, optimizer_state = _train_task(
            model, task_id, train_loader, val_loader, config, device, mean_lr_multipliers
        )
        mean_lr_multipliers = optimizer_state["mean_lr_multipliers"]
        checkpoint_path = run_dir / f"model_after_task_{task_id:02d}.pt"
        _checkpoint(checkpoint_path, model, task_id, train_starts, config, optimizer_state, best, history)
        print(f"Saved task {task_id} checkpoint: {checkpoint_path}")
        if args.evaluate_after_task or config.get("ucb_evaluate_after_task", False):
            _evaluate_task_checkpoint(_ROOT, run_dir, task_id, checkpoint_path, config)


if __name__ == "__main__":
    main()
