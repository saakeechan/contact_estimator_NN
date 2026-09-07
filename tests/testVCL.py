"""VCL Bayesian-TCN hooks for the shared probabilistic velocity evaluator."""

import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch

from contact_cnn import ContactCNNWithNormalization, VCLTCN
from tests.base_test import ProbabilisticVelocitySingleTest


class VCLSingleTest(ProbabilisticVelocitySingleTest):
    default_seed = 203
    default_cmd_vel_x_window = (0.0, 3.0)
    model_config_name = "VCL_params.yaml"
    find_trajectory = staticmethod(ProbabilisticVelocitySingleTest.find_random_trajectory)

    def add_model_arguments(self, parser):
        parser.add_argument("--knn-cache", help="Optional cache for the active VCL latent reference.")
        parser.add_argument("--skip-plots", action="store_true", help="Do not save the per-trajectory velocity plot.")
        parser.add_argument("--mc-samples", type=int, help="Override vcl_evaluation_mc_samples from the config.")

    def set_runtime_values(self, args):
        self.mc_samples_override = args.mc_samples

    def configure(self, config):
        samples = int(getattr(self, "mc_samples_override", None) or config["vcl_evaluation_mc_samples"])
        if samples < 2:
            raise ValueError("VCL predictive uncertainty requires at least two samples.")
        config["vcl_evaluation_mc_samples"] = samples
        return config

    def build_model(self, config, num_features):
        if config.get("model_architecture", "tcn").lower() != "tcn":
            raise ValueError("VCL inference requires model_architecture: 'tcn'.")
        return ContactCNNWithNormalization(VCLTCN(
            config["window_size"], num_features, config.get("tcn_num_channels", 64),
            config.get("tcn_kernel_size", 3), config.get("tcn_num_blocks", 5), config.get("tcn_dropout", .2),
            config.get("vcl_rho", -3.0), vcl_prior_sigma=float(config.get("vcl_prior_sigma", 1.0)),
        ))

    def find_checkpoint(self, num_features, config):
        logs_root = _ROOT / "logs" / "logsVCL"
        for run_dir in sorted(glob.glob(str(logs_root / "run_*")), key=os.path.getmtime, reverse=True):
            checkpoints = sorted(glob.glob(os.path.join(run_dir, "task_*_posterior.pt")), reverse=True)
            for checkpoint in checkpoints:
                state = torch.load(checkpoint, map_location="cpu")
                weights = state["model_state_dict"]
                if weights["base_model.input_proj.conv.weight_mu"].shape[1] == num_features:
                    return checkpoint
        raise FileNotFoundError(f"No VCL checkpoint with {num_features} input features found in {logs_root}.")

    @staticmethod
    @torch.no_grad()
    def stochastic_outputs(model, windows, samples):
        model.eval()
        outputs = [model(windows, return_sequence=False, sample=True) for _ in range(samples)]
        means = torch.stack([output[1][:, 0] for output in outputs])
        aleatoric = torch.stack([output[3][:, 0] for output in outputs]).mean(dim=0)
        return means.mean(dim=0), aleatoric, means.var(dim=0, unbiased=False)

    def run_trajectory(self, model, trajectory, window_size, batch_size, device):
        features = self.make_features(trajectory)
        predictions, aleatoric, epistemic = [], [], []
        for first in range(0, len(features) - window_size + 1, batch_size):
            last = min(first + batch_size, len(features) - window_size + 1)
            windows = torch.from_numpy(np.stack([features[index:index + window_size] for index in range(first, last)])).float().to(device)
            mean, noise, disagreement = self.stochastic_outputs(model, windows, self.mc_samples)
            predictions.append(mean.cpu().numpy())
            aleatoric.append(noise.cpu().numpy())
            epistemic.append(disagreement.cpu().numpy())
        final_indices = np.arange(window_size - 1, len(trajectory))
        # The base evaluator only needs the total predictive variance here.
        total = np.concatenate(aleatoric) + np.concatenate(epistemic)
        return (final_indices, np.concatenate(predictions), total, np.zeros(len(final_indices)),
                trajectory["lfoot-contact"].to_numpy()[final_indices], self.make_body_velocity(trajectory)[final_indices])

    @staticmethod
    def get_training_window_starts(data_folder, window_size, config):
        starts = config.get("vcl_training_window_starts")
        if starts is not None:
            return np.asarray(starts, dtype=np.int64)
        return ProbabilisticVelocitySingleTest.get_training_window_starts(data_folder, window_size, config)

    @staticmethod
    @torch.no_grad()
    def collect_final_tcn_latents(model, raw_data, window_starts, window_size, batch_size, device):
        offsets, latents = np.arange(window_size), []
        model.eval()
        for first in range(0, len(window_starts), batch_size):
            starts = window_starts[first:first + batch_size]
            windows = torch.from_numpy(raw_data[starts[:, None] + offsets]).float().to(device)
            normalized = (windows - model.global_mean) / (model.global_std + model.eps)
            latents.append(model.base_model.extract_features(normalized)[:, :, -1].cpu())
        return torch.cat(latents).numpy()

    def uncertainty_for_window(self, model, features, window_start, window_size, device):
        window = torch.from_numpy(features[window_start:window_start + window_size]).float().unsqueeze(0).to(device)
        _, aleatoric, epistemic = self.stochastic_outputs(model, window, self.mc_samples)
        return aleatoric.squeeze(0).cpu().numpy(), epistemic.squeeze(0).cpu().numpy()

    def evaluate(self, args, context, model=None, checkpoint_path=None):
        self.mc_samples = int(context["config"]["vcl_evaluation_mc_samples"])
        if checkpoint_path is None:
            features = self.make_features(context["trajectory"])
            checkpoint_path = self.find_checkpoint(features.shape[1], context["config"])
        checkpoint = torch.load(checkpoint_path, map_location=context["device"])
        context["config"]["vcl_training_window_starts"] = checkpoint.get("train_window_starts")
        if model is None:
            features = self.make_features(context["trajectory"])
            model = self.build_model(context["config"], features.shape[1]).to(context["device"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        metrics, model, checkpoint_path = super().evaluate(args, context, model, checkpoint_path)
        aleatoric, epistemic = np.asarray(metrics["aleatoric_variance"]), np.asarray(metrics["epistemic_variance"])
        metrics["vcl_mc_samples"] = self.mc_samples
        metrics["total_variance"] = (aleatoric + epistemic).tolist()
        return metrics, model, checkpoint_path


def run_evaluation(args, context, model=None, checkpoint_path=None):
    return VCLSingleTest().evaluate(args, context, model, checkpoint_path)


if __name__ == "__main__":
    VCLSingleTest().main()
