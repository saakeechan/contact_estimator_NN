"""MC-dropout checkpoint lookup for MuJoCo environment replay."""
import glob
import os
from pathlib import Path

import torch

from tests.testMCdropout import MCDropoutSingleTest

_ROOT = Path(__file__).resolve().parents[1]


class ReplayMCDropoutMujocoSingleTest(MCDropoutSingleTest):
    model_config_name = 'ReplayMCdropoutMujoco_params.yaml'

    def find_checkpoint(self, num_features, config):
        logs_root = _ROOT / 'logs' / 'logsReplayMCDropoutMujoco'
        for run_dir in sorted(glob.glob(str(logs_root / 'run_*')), key=os.path.getmtime, reverse=True):
            for checkpoint in sorted(glob.glob(str(Path(run_dir) / 'model_after_task_*.pt')), reverse=True):
                state = torch.load(checkpoint, map_location='cpu')['model_state_dict']
                if state['base_model.input_proj.weight'].shape[1] == num_features:
                    return checkpoint
        raise FileNotFoundError(f'No MuJoCo replay MC-dropout checkpoint with {num_features} input features found in {logs_root}.')
