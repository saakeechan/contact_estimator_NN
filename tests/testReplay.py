"""Deterministic replay-TCN hooks for the shared single-trajectory evaluator."""

import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch

from contact_cnn import ContactCNNWithNormalization, EnsembleTCN
from tests.base_test import ProbabilisticVelocitySingleTest


class ReplaySingleTest(ProbabilisticVelocitySingleTest):
    default_seed = 203
    default_cmd_vel_x_window = (0.0, 3.0)
    model_config_name = "Replay_params.yaml"
    find_trajectory = staticmethod(ProbabilisticVelocitySingleTest.find_random_trajectory)

    def add_model_arguments(self, parser):
        parser.add_argument("--checkpoint-path", help="Replay model_after_task_XX.pt to evaluate.")
        parser.add_argument("--knn-cache", help="Optional cache for the checkpoint's replay-latent reference.")
        parser.add_argument("--skip-plots", action="store_true", help="Do not save the velocity plot.")

    def set_runtime_values(self, args):
        self.checkpoint_override = getattr(args, "checkpoint_path", None)

    def build_model(self, config, num_features):
        if config.get("model_architecture", "tcn").lower() != "tcn":
            raise ValueError("Replay inference requires model_architecture: 'tcn'.")
        return ContactCNNWithNormalization(EnsembleTCN(
            config["window_size"], num_features, config.get("tcn_num_channels", 64),
            config.get("tcn_kernel_size", 3), config.get("tcn_num_blocks", 5), config.get("tcn_dropout", .2),
        ))

    def find_checkpoint(self, num_features, config):
        if getattr(self, "checkpoint_override", None):
            checkpoint = self.checkpoint_override
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(checkpoint)
            return checkpoint
        for run_dir in sorted(glob.glob(str(_ROOT / "logsReplay" / "run_*")), key=os.path.getmtime, reverse=True):
            checkpoints = sorted(glob.glob(os.path.join(run_dir, "model_after_task_*.pt")), reverse=True)
            for checkpoint in checkpoints:
                state = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
                if state["base_model.input_proj.weight"].shape[1] == num_features:
                    return checkpoint
        raise FileNotFoundError(f"No replay checkpoint with {num_features} input features found in {_ROOT / 'logsReplay'}.")

    @staticmethod
    def get_training_window_starts(data_folder, window_size, config):
        starts = config.get("replay_training_window_starts")
        if starts is None:
            return ProbabilisticVelocitySingleTest.get_training_window_starts(data_folder, window_size, config)
        return np.asarray(starts, dtype=np.int64)

    @staticmethod
    def uncertainty_for_window(model, features, window_start, window_size, device):
        window = torch.from_numpy(features[window_start:window_start + window_size]).float().unsqueeze(0).to(device)
        _, _, _, variance, _ = model(window, return_sequence=False)
        aleatoric = variance[0, 0].detach().cpu().numpy()
        return aleatoric, np.zeros_like(aleatoric)

    def evaluate(self, args, context, model=None, checkpoint_path=None):
        if checkpoint_path is None and getattr(self, "checkpoint_override", None):
            checkpoint_path = self.checkpoint_override
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            starts = checkpoint.get("replay_window_starts")
            if starts is None or len(starts) == 0:
                starts = np.concatenate(checkpoint.get("task_train_window_starts", [])).tolist()
            context["config"]["replay_training_window_starts"] = starts
        return super().evaluate(args, context, model, checkpoint_path)


def run_evaluation(args, context, model=None, checkpoint_path=None):
    return ReplaySingleTest().evaluate(args, context, model, checkpoint_path)


if __name__ == "__main__":
    ReplaySingleTest().main()
