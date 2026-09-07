"""Train the shared-head Bayesian TCN with Variational Continual Learning."""

import argparse
import copy
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
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from normalization import ContactCNNWithNormalization
from vcl import VCLTCN
from common import resolve_active_legs
from utils.ucb_task_windows import _validate_task_windows


class TaskWindowDataset(Dataset):
    """Raw, run-safe windows selected by the precomputed continual task index."""
    def __init__(self, data, labels, velocities, starts, window_size):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.velocities = torch.as_tensor(velocities, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window_size = int(window_size)
        if not len(self.starts):
            raise ValueError("A VCL task split has no windows.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start, end = int(self.starts[index]), int(self.starts[index]) + self.window_size
        return {"data": self.data[start:end], "label": self.labels[end - 1],
                "label_seq": self.labels[start:end], "velocity": self.velocities[start:end]}


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _load_config(root, network_config, vcl_config):
    with network_config.open() as file:
        config = yaml.safe_load(file) or {}
    with vcl_config.open() as file:
        config.update(yaml.safe_load(file) or {})
    config["data_folder"] = str(_resolve(root, config["data_folder"]))
    return config


def _load_task_index(index_path, task_windows, window_size):
    with np.load(index_path) as index:
        required = {"task_windows", "window_starts", "task_ids", "window_run_ids"}
        missing = required.difference(index.files)
        if missing:
            raise ValueError(f"Task index is missing: {sorted(missing)}. Rebuild it with utils/ucb_task_windows.py.")
        indexed_windows = index["task_windows"]
        if not np.array_equal(indexed_windows, task_windows):
            if len(task_windows) > len(indexed_windows) or not np.array_equal(indexed_windows[:len(task_windows)], task_windows):
                raise ValueError("Task index windows do not match VCL_params.yaml; rebuild the task index.")
        if "window_size" in index and int(index["window_size"]) != int(window_size):
            raise ValueError("Task index window size does not match the active network config; rebuild the task index.")
        values = {name: index[name].copy() for name in required}
        selected = values["task_ids"] < len(task_windows)
        return {name: value[selected] if name != "task_windows" else value for name, value in values.items()}


def _split_task_windows(index, task_id, train_ratio, data_fraction, seed):
    mask = index["task_ids"] == task_id
    starts, run_ids = index["window_starts"][mask], index["window_run_ids"][mask]
    if not 0 < data_fraction <= 1:
        raise ValueError("vcl_data_fraction must be in (0, 1].")
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
    flattened = torch.as_tensor(data[starts[:, None] + offsets], dtype=torch.float32).reshape(-1, data.shape[-1])
    low, high = torch.quantile(flattened, .01, dim=0), torch.quantile(flattened, .99, dim=0)
    clipped = flattened.clamp(low, high)
    return (clipped.mean(dim=0).view(1, 1, -1).to(device),
            clipped.std(dim=0).clamp_min(1e-8).view(1, 1, -1).to(device))


def _make_loaders(data, labels, velocities, train_starts, val_starts, config, seed):
    train = TaskWindowDataset(data, labels, velocities, train_starts, config["window_size"])
    val = TaskWindowDataset(data, labels, velocities, val_starts, config["window_size"])
    return (DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed)),
            DataLoader(val, batch_size=config.get("test_batch_size", config["batch_size"]), shuffle=False))


def _likelihood_nll(outputs, sample, dense, contact_weight, velocity_weight, reduction="mean"):
    velocity_target = sample["velocity"] if dense else sample["velocity"][:, -1]
    contact_target = sample["label_seq"] if dense else sample["label"]
    velocity_prediction, variance = ((outputs[0].permute(0, 3, 1, 2), outputs[2].permute(0, 3, 1, 2))
                                     if dense else (outputs[1], outputs[3]))
    contact_mask = (contact_target == 1).float().unsqueeze(-1)
    velocity_nll = (F.gaussian_nll_loss(velocity_prediction, velocity_target, variance, reduction="none") * contact_mask).sum()
    contact_nll = F.binary_cross_entropy_with_logits(outputs[4], sample["label"], reduction="sum")
    if reduction == "mean":
        velocity_nll = velocity_nll / (contact_mask.sum() * velocity_target.shape[-1] + 1e-8)
        contact_nll = contact_nll / sample["label"].numel()
    elif reduction != "sum":
        raise ValueError("reduction must be 'mean' or 'sum'.")
    return contact_weight * contact_nll + velocity_weight * velocity_nll, contact_nll, velocity_nll


@torch.no_grad()
def _predictive_outputs(model, inputs, dense, samples):
    outputs = [model(inputs, return_sequence=dense, sample=True) for _ in range(samples)]
    indices = (0, 2) if dense else (1, 3)
    mean = torch.stack([output[indices[0]] for output in outputs]).mean(dim=0)
    aleatoric = torch.stack([output[indices[1]] for output in outputs]).mean(dim=0)
    epistemic = torch.stack([output[indices[0]] for output in outputs]).var(dim=0, unbiased=False)
    contact = torch.stack([output[4] for output in outputs]).mean(dim=0)
    total_variance = aleatoric + epistemic
    return (mean if dense else None, mean[..., -1] if dense else mean,
            total_variance if dense else None, total_variance[..., -1] if dense else total_variance,
            contact), aleatoric, epistemic


@torch.no_grad()
def _evaluate(model, loader, device, config):
    model.eval()
    dense, samples = config.get("use_dense_supervision", False), int(config["vcl_evaluation_mc_samples"])
    totals = dict(loss=0.0, contact_loss=0.0, velocity_loss=0.0, absolute=0.0, squared=0.0,
                  valid=0.0, aleatoric=0.0, epistemic=0.0, total_variance=0.0)
    for sample in loader:
        sample = {name: value.to(device) for name, value in sample.items()}
        outputs, aleatoric, epistemic = _predictive_outputs(model, sample["data"], dense, samples)
        loss, contact_loss, velocity_loss = _likelihood_nll(outputs, sample, dense,
            float(config.get("contact_weight", 1.0)), float(config.get("velocity_weight", 1.0)))
        prediction, target = ((outputs[0].permute(0, 3, 1, 2), sample["velocity"])
                              if dense else (outputs[1], sample["velocity"][:, -1]))
        mask = ((sample["label_seq"] if dense else sample["label"]) == 1).float().unsqueeze(-1)
        count = (mask.sum() * target.shape[-1]).item()
        totals["loss"] += loss.item(); totals["contact_loss"] += contact_loss.item(); totals["velocity_loss"] += velocity_loss.item()
        totals["absolute"] += ((prediction - target).abs() * mask).sum().item()
        totals["squared"] += ((prediction - target).square() * mask).sum().item()
        totals["aleatoric"] += (aleatoric * mask).sum().item(); totals["epistemic"] += (epistemic * mask).sum().item()
        totals["total_variance"] += ((aleatoric + epistemic) * mask).sum().item(); totals["valid"] += count
    batches, valid = max(1, len(loader)), max(1.0, totals["valid"])
    return {"loss": totals["loss"] / batches, "contact_loss": totals["contact_loss"] / batches,
            "velocity_loss": totals["velocity_loss"] / batches, "velocity_mae": totals["absolute"] / valid,
            "velocity_rmse": math.sqrt(totals["squared"] / valid), "mean_aleatoric_variance": totals["aleatoric"] / valid,
            "mean_epistemic_variance": totals["epistemic"] / valid,
            "mean_total_variance": totals["total_variance"] / valid,
            "total_variance_calibration_ratio": totals["squared"] / (totals["total_variance"] + 1e-8)}


def _posterior_diagnostics(model, reference=None):
    result = {}
    for name, layer in model.base_model.named_modules():
        if not hasattr(layer, "weight_mu") or not hasattr(layer, "posterior_sigma"):
            continue
        sigma = layer.posterior_sigma().detach().flatten()
        result[name] = {"sigma_quantiles": torch.quantile(sigma, torch.tensor([.05, .5, .95], device=sigma.device)).cpu().tolist(),
                        "mean_abs_mu_movement": 0.0 if reference is None else (layer.weight_mu.detach() - reference[name]).abs().mean().item()}
    return result


def _train_task(model, task_id, train_loader, val_loader, config, device):
    """Optimize q_t against the task-start prior q_(t-1), which stays frozen."""
    optimizer = Adam(model.parameters(), lr=float(config.get("vcl_init_lr", config["init_lr"])))
    samples, dataset_size = int(config["vcl_train_mc_samples"]), len(train_loader.dataset)
    if samples < 1 or dataset_size < 1:
        raise ValueError("VCL requires at least one Monte-Carlo sample and one training window.")
    dense, scale = config.get("use_dense_supervision", False), float(config.get("vcl_kl_scale", 1.0))
    patience_limit, patience, lr_updates = int(config.get("vcl_lr_patience", 5)), int(config.get("vcl_lr_patience", 5)), 0
    best, history = {"loss": float("inf"), "state": None, "epoch": -1}, []
    reference = {name: layer.weight_mu.detach().clone() for name, layer in model.base_model.named_modules()
                 if hasattr(layer, "weight_mu")}
    for epoch in range(int(config["num_epoch"])):
        model.train(); totals = dict(loss=0.0, nll=0.0, kl=0.0, mu_grad=0.0, rho_grad=0.0)
        progress = tqdm(train_loader, desc=f"Task {task_id} epoch {epoch + 1}/{config['num_epoch']}", unit="batch")
        for sample in progress:
            sample = {name: value.to(device) for name, value in sample.items()}
            nll = torch.stack([_likelihood_nll(model(sample["data"], return_sequence=dense, sample=True), sample, dense,
                float(config.get("contact_weight", 1.0)), float(config.get("velocity_weight", 1.0)))[0] for _ in range(samples)]).mean()
            kl = model.base_model.kl_to_prior()
            loss = nll + scale * kl / dataset_size
            optimizer.zero_grad(set_to_none=True); loss.backward()
            totals["mu_grad"] += sum(parameter.grad.norm().item() for name, parameter in model.named_parameters()
                                     if name.endswith("_mu") and parameter.grad is not None)
            totals["rho_grad"] += sum(parameter.grad.norm().item() for name, parameter in model.named_parameters()
                                      if name.endswith("_rho") and parameter.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("vcl_gradient_clip_norm", 10.0)))
            optimizer.step()
            totals["loss"] += loss.item(); totals["nll"] += nll.item(); totals["kl"] += kl.item()
            progress.set_postfix(loss=f"{loss.item():.3g}", nll=f"{nll.item():.3g}", kl=f"{(scale * kl / dataset_size).item():.3g}")
        validation = _evaluate(model, val_loader, device, config)
        batches = len(train_loader)
        metrics = {"loss": totals["loss"] / batches, "expected_nll": totals["nll"] / batches,
                   "kl_total": totals["kl"] / batches, "kl_per_example": scale * totals["kl"] / batches / dataset_size,
                   "mu_gradient_norm": totals["mu_grad"] / batches, "rho_gradient_norm": totals["rho_grad"] / batches}
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], **metrics,
                        **{f"val_{key}": value for key, value in validation.items()}})
        print(f"Task {task_id} epoch {epoch + 1}/{config['num_epoch']} | train VCL {metrics['loss']:.5f} | val NLL {validation['loss']:.5f} | val MAE {validation['velocity_mae']:.5f}")
        if validation["loss"] < best["loss"]:
            best = {"loss": validation["loss"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch}; patience = patience_limit
        else:
            patience -= 1
            if patience <= 0:
                for group in optimizer.param_groups:
                    group["lr"] /= float(config.get("vcl_lr_factor", 3.0))
                lr_updates += 1; patience = patience_limit
                if optimizer.param_groups[0]["lr"] < float(config.get("vcl_min_lr", 1e-6)) or lr_updates > int(config.get("vcl_max_lr_updates", 3)):
                    break
    model.load_state_dict(best["state"])
    return best, history, optimizer.state_dict(), _posterior_diagnostics(model, reference)


def _checkpoint(path, model, task_id, train_starts, config, optimizer_state, best, history, task_metrics, prior_advanced):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer_state, "task_id": task_id,
                "next_task_id": task_id + 1, "prior_advanced": prior_advanced,
                "train_window_starts": np.asarray(train_starts).tolist(), "task_dataset_size": len(train_starts),
                "vcl_kl_scale": float(config.get("vcl_kl_scale", 1.0)), "vcl_scaling": "mean_batch_nll + kl_scale * KL / N_t",
                "vcl_config": config, "random_seed": int(config.get("random_seed", 42)), "best_validation_loss": best["loss"],
                "best_epoch": best["epoch"], "history": history, "task_metrics": task_metrics,
                "result_matrix": task_metrics["result_matrix"]}, path)


def _validate_resume_checkpoint(checkpoint):
    if not checkpoint.get("prior_advanced", False):
        raise ValueError("Resume from task_NN_propagation.pt: posterior checkpoints retain the old-task prior for audit.")


def _evaluate_task_checkpoint(run_dir, task_id, checkpoint_path, config):
    """Run the shared VCL series evaluator across the configured evaluation window."""
    results_dir = _ROOT / "testResults" / "VCL" / run_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / f"vcl_series_after_task_{task_id:02d}.csv"
    subprocess.run([
        sys.executable, _ROOT / "tests/base_testSeries.py", "--model", "vcl",
        "--checkpoint-path", checkpoint_path, "--cmd-vel-x-window",
        str(config["evaluation_window"][0]), str(config["evaluation_window"][1]),
        "--first-seed", str(config.get("vcl_evaluation_first_seed", 0)),
        "--last-seed", str(config.get("vcl_evaluation_last_seed", 100)),
        "--output-csv", output_csv, "--save-csv", "--plot-metric", "epistemic",
        "--skip-knn", "--quiet", "--fail-if-output-exists",
    ], cwd=_ROOT, check=True)


@torch.no_grad()
def _assert_prior_matches_previous_posterior(model):
    for name, layer in model.base_model.named_modules():
        if not hasattr(layer, "weight_mu"):
            continue
        if not (torch.allclose(layer.prior_weight_mu, layer.weight_mu)
                and torch.allclose(layer.prior_weight_sigma, layer.posterior_sigma())):
            raise RuntimeError(f"Task-start VCL prior for {name} does not match the saved previous posterior.")
        if layer.bias_mu is not None and not (torch.allclose(layer.prior_bias_mu, layer.bias_mu)
                                             and torch.allclose(layer.prior_bias_sigma, layer.posterior_scale(layer.bias_rho))):
            raise RuntimeError(f"Task-start VCL bias prior for {name} does not match the saved previous posterior.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=Path, default=_ROOT / "config/network_params.yaml")
    parser.add_argument("--vcl-config", type=Path, default=_ROOT / "config/VCL_params.yaml")
    parser.add_argument("--task-index", type=Path, default=_ROOT / "Data/NumpyFiles/ucb_task_windows.npz")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume from a completed task_NN_propagation.pt checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print deterministic task splits without training.")
    parser.add_argument("--skip-evaluate-after-task", action="store_true", help="Do not run the VCL test series after each task.")
    args = parser.parse_args()
    config = _load_config(_ROOT, args.config_name, args.vcl_config)
    task_windows = _validate_task_windows(config["continual_task_windows"])
    index = _load_task_index(args.task_index, task_windows, config["window_size"])
    data_folder = Path(config["data_folder"])
    data, labels, velocities = (np.load(data_folder / name) for name in ("all_data.npy", "all_labels.npy", "all_body_velocities.npy"))
    metadata = np.load(data_folder / 'all_data_metadata.npy', allow_pickle=True).item()
    legs = tuple(metadata.get('legs', ()))
    if legs != resolve_active_legs(config.get('active_legs', 'both')) or labels.shape != (len(data), len(legs)) or velocities.shape != (len(data), len(legs), 3):
        raise ValueError('Dataset legs/shapes do not match active_legs. Regenerate the numpy dataset and task index.')
    config['legs'] = legs
    starts = index["window_starts"]
    if np.any(starts < 0) or np.any(starts + config["window_size"] > len(data)):
        raise ValueError("Task index contains starts outside all_data.npy; rebuild the task index.")
    seed, fraction = int(config.get("random_seed", 42)), float(config.get("vcl_data_fraction", 1.0))
    splits = [_split_task_windows(index, task_id, float(config.get("train_ratio", .85)), fraction, seed)
              for task_id in range(len(task_windows))]
    for task_id, (train_starts, val_starts) in enumerate(splits):
        print(f"Task {task_id} [{task_windows[task_id, 0]:g}, {task_windows[task_id, 1]:g}): {len(train_starts)} train, {len(val_starts)} validation windows")
    if args.dry_run:
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, start_task = None, 0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device); _validate_resume_checkpoint(checkpoint)
        start_task, run_dir = int(checkpoint["next_task_id"]), args.resume_checkpoint.resolve().parent
    else:
        run_dir = _ROOT / "logs" / "logsVCL" / f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        run_dir.mkdir(parents=True, exist_ok=False)
        with (run_dir / "network_params.yaml").open("w") as file:
            yaml.safe_dump(config, file, sort_keys=False)
    if start_task >= len(task_windows):
        raise ValueError("The resume checkpoint already completed every configured task.")
    mean, std = _normalization_stats(data, splits[0][0], config["window_size"], device)
    model = ContactCNNWithNormalization(VCLTCN(
        config["window_size"], data.shape[1], config.get("tcn_num_channels", 64), config.get("tcn_kernel_size", 3),
        config.get("tcn_num_blocks", 5), config.get("tcn_dropout", .2), config.get("vcl_rho", -3.0),
        vcl_prior_sigma=float(config.get("vcl_prior_sigma", 1.0)), legs=config['legs']), global_mean=mean, global_std=std).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        print(f"Initial posterior sigma quantiles: {_posterior_diagnostics(model)}")
    result_matrix = [] if checkpoint is None else checkpoint.get("result_matrix", checkpoint.get("task_metrics", {}).get("result_matrix", []))
    for task_id in range(start_task, len(task_windows)):
        if task_id > 0:
            _assert_prior_matches_previous_posterior(model)
        train_starts, val_starts = splits[task_id]
        train_loader, val_loader = _make_loaders(data, labels, velocities, train_starts, val_starts, config, seed + task_id)
        best, history, optimizer_state, diagnostics = _train_task(model, task_id, train_loader, val_loader, config, device)
        seen_metrics = {str(domain): _evaluate(model, _make_loaders(data, labels, velocities, *splits[domain], config, seed + domain)[1], device, config)
                        for domain in range(task_id + 1)}
        result_matrix.append(seen_metrics)
        for domain, metrics in seen_metrics.items():
            previous = [row.get(domain, {}).get("velocity_mae", float("inf")) for row in result_matrix[:-1]]
            metrics["forgetting_velocity_mae"] = metrics["velocity_mae"] - min(previous, default=metrics["velocity_mae"])
        task_metrics = {"seen_domains": seen_metrics, "posterior_diagnostics": diagnostics, "result_matrix": result_matrix}
        posterior_path = run_dir / f"task_{task_id:02d}_posterior.pt"
        _checkpoint(posterior_path, model, task_id, train_starts, config, optimizer_state, best, history, task_metrics, False)
        model.base_model.set_prior_from_posterior()
        propagation_path = run_dir / f"task_{task_id:02d}_propagation.pt"
        _checkpoint(propagation_path, model, task_id, train_starts, config, optimizer_state, best, history, task_metrics, True)
        print(f"Saved posterior audit checkpoint: {posterior_path}\nSaved next-task propagation checkpoint: {propagation_path}")
        if config.get("vcl_evaluate_after_task", True) and not args.skip_evaluate_after_task:
            _evaluate_task_checkpoint(run_dir, task_id, posterior_path, config)


if __name__ == "__main__":
    main()
