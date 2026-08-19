"""Shared command-line setup for model-specific single-trajectory tests."""
import argparse
import os

import numpy as np
import torch
import yaml

from utils.ood_selection import csv_matches_environment_windows, validate_ood_selection


class BaseSingleTest:
    """Select one evaluation trajectory, then delegate inference to a model subclass."""

    description = 'Run one selected CSV trajectory through the contact network.'
    default_cmd_vel_x_window = (2.0, 3.0)
    model_config_name = None

    def build_parser(self):
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
        parser.add_argument('--seed', type=int, default=self.default_seed)
        parser.add_argument('--cmd-vel-x-window', type=float, nargs=2, metavar=('MIN', 'MAX'),
                            default=self.default_cmd_vel_x_window)
        parser.add_argument('--ood-feature', choices=('cmd_vel', 'environment'), help='Override ood_feature from the config.')
        parser.add_argument('--environment-window', type=int, nargs=2, action='append', metavar=('MIN', 'MAX'),
                            help='Inclusive environment window; repeat this option for disjoint windows.')
        parser.add_argument('--metrics-json', help='Optional path for machine-readable evaluation metrics.')
        parser.add_argument('--skip-umap', action='store_true', help='Compute kNN/OOD metrics without saving a UMAP figure.')
        parser.add_argument('--save-umap', action='store_true', help='Save the kNN UMAP figure (disabled by default).')
        parser.add_argument('--skip-knn', action='store_true', help='Do not run kNN/OOD evaluation.')
        self.add_model_arguments(parser)
        return parser

    def add_model_arguments(self, parser):
        """Add arguments used only by a concrete model evaluator."""

    def load_config(self, config_name):
        config = {}
        if self.model_config_name:
            model_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', self.model_config_name)
            with open(model_config_path) as config_file:
                config.update(yaml.safe_load(config_file) or {})
        with open(config_name) as config_file:
            config.update(yaml.safe_load(config_file) or {})
        return self.configure(config)

    def configure(self, config):
        return config

    def prepare(self, args):
        self.set_runtime_values(args)
        config = self.load_config(args.config_name)
        ood_feature = args.ood_feature or config.get('ood_feature', 'cmd_vel')
        environment_windows = args.environment_window or config.get('environment_windows', [])
        validate_ood_selection(ood_feature, environment_windows)
        cmd_vel_x_window = tuple(args.cmd_vel_x_window)
        low, high = cmd_vel_x_window
        if low > high:
            raise ValueError('TEST_CMD_VEL_X_WINDOW must be (min, max) with min <= max')

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_folder = config['csv_folder'] if os.path.isabs(config['csv_folder']) else os.path.join(project_root, config['csv_folder'])
        trajectory, csv_path, run_index, start_cmd_vel = self.find_trajectory(
            csv_folder, config['window_size'], cmd_vel_x_window, np.random.default_rng(args.seed),
            ood_feature, environment_windows,
        )
        environment_id = csv_matches_environment_windows(csv_path, environment_windows)[1] if ood_feature == 'environment' else None
        return {
            'config': config, 'ood_feature': ood_feature, 'environment_windows': environment_windows,
            'cmd_vel_x_window': cmd_vel_x_window, 'project_root': project_root,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'trajectory': trajectory, 'csv_path': csv_path, 'run_index': run_index,
            'start_cmd_vel': start_cmd_vel, 'environment_id': environment_id,
        }

    def main(self):
        args = self.build_parser().parse_args()
        return self.run(args, self.prepare(args))

    def set_runtime_values(self, args):
        """Update legacy module-level defaults used by plotting helpers."""

    def find_trajectory(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, args, context):
        raise NotImplementedError
