"""Train an MC-dropout TCN task-by-task with balanced replay."""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from contact_cnn import ContactCNNWithNormalization, MCDropoutTCN
from train.trainMCdropout import MCDropoutTrainer
from train.trainReplay import _load_config, run_replay_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=Path, default=_ROOT / "config/network_params.yaml")
    parser.add_argument("--replay-mc-config", type=Path, default=_ROOT / "config/ReplayMCdropout_params.yaml")
    parser.add_argument("--task-index", type=Path,
                        default=_ROOT / "Data/NumpyFiles/replay_mc_dropout_task_windows.npz")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-evaluate-after-task", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config_name, args.replay_mc_config)
    if config.get("model_architecture", "tcn").lower() != "tcn":
        raise ValueError("Replay MC-dropout training requires model_architecture: 'tcn'.")
    if not 0.0 < float(config["mc_dropout_rate"]) < 1.0 or int(config["mc_dropout_samples"]) < 2:
        raise ValueError("mc_dropout_rate must be in (0, 1) and mc_dropout_samples must be at least 2.")
    run_replay_training(
        config, args.task_index,
        lambda cfg, features, mean, std: ContactCNNWithNormalization(MCDropoutTCN(
            cfg["window_size"], features, cfg.get("tcn_num_channels", 64), cfg.get("tcn_kernel_size", 3),
            cfg.get("tcn_num_blocks", 5), cfg["mc_dropout_rate"]), mean, std),
        MCDropoutTrainer, "logsReplayMCDropout", "replay_mc_dropout", "ReplayMCDropout", "epistemic",
        args.dry_run, args.skip_evaluate_after_task,
    )


if __name__ == "__main__":
    main()
