"""Build the combined replay/MC-dropout task index from its own YAML config."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from utils.replay_task_windows import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--replay-config", str(_ROOT / "config/ReplayMCdropout_params.yaml"),
        "--output", str(_ROOT / "Data/NumpyFiles/replay_mc_dropout_task_windows.npz"),
    ]
    main()
