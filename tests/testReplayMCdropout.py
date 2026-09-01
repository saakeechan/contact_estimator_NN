"""Replay-checkpoint hooks for the existing MC-dropout evaluator."""

import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from tests.testMCdropout import MCDropoutSingleTest


class ReplayMCDropoutSingleTest(MCDropoutSingleTest):
    model_config_name = "ReplayMCdropout_params.yaml"

    def add_model_arguments(self, parser):
        super().add_model_arguments(parser)
        parser.add_argument("--checkpoint-path", help="Replay MC-dropout model_after_task_XX.pt to evaluate.")

    def set_runtime_values(self, args):
        super().set_runtime_values(args)
        self.checkpoint_override = getattr(args, "checkpoint_path", None)

    def find_checkpoint(self, num_features, config):
        if getattr(self, "checkpoint_override", None):
            return self.checkpoint_override
        for run_dir in sorted(glob.glob(str(_ROOT / "logsReplayMCDropout" / "run_*")), key=os.path.getmtime, reverse=True):
            for checkpoint in sorted(glob.glob(os.path.join(run_dir, "model_after_task_*.pt")), reverse=True):
                state = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
                if state["base_model.input_proj.weight"].shape[1] == num_features:
                    return checkpoint
        raise FileNotFoundError(f"No replay MC-dropout checkpoint with {num_features} input features found.")

    @staticmethod
    def get_training_window_starts(data_folder, window_size, config):
        starts = config.get("replay_training_window_starts")
        if starts is None or len(starts) == 0:
            starts = MCDropoutSingleTest.get_training_window_starts(data_folder, window_size, config)
        return np.asarray(starts, dtype=np.int64)

    def evaluate(self, args, context, model=None, checkpoint_path=None):
        checkpoint_path = checkpoint_path or getattr(self, "checkpoint_override", None)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            starts = checkpoint.get("replay_window_starts")
            if starts is None or len(starts) == 0:
                starts = np.concatenate(checkpoint["task_train_window_starts"]).tolist()
            context["config"]["replay_training_window_starts"] = starts
            self.checkpoint_override = checkpoint_path
        return super().evaluate(args, context, model, checkpoint_path)


def run_evaluation(args, context, model=None, checkpoint_path=None):
    return ReplayMCDropoutSingleTest().evaluate(args, context, model, checkpoint_path)


if __name__ == "__main__":
    ReplayMCDropoutSingleTest().main()
