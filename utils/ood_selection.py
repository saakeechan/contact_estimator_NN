"""Shared filename-based OOD selection helpers for CSV conversion and evaluation."""

import os
import re


def environment_number_from_filename(data_name):
    """Return the environment ID encoded in a CSV filename, or ``None``.

    Accepted forms include ``robotstate_0.csv``, ``run_env5.csv``,
    ``run_environment_12.csv``, and ``run_env(8).csv``.
    """
    stem = os.path.splitext(os.path.basename(data_name))[0]
    match = re.search(r'(?:^|_)(?:env|environment)_?\(?(\d+)\)?(?:_|$)', stem, re.IGNORECASE)
    if match is None:
        match = re.search(r'_(\d+)$', stem)
    return int(match.group(1)) if match else None


def validate_ood_selection(ood_feature, environment_windows):
    """Validate the config values used by CSV evaluation selection."""
    if ood_feature not in ('cmd_vel', 'environment'):
        raise ValueError("ood_feature must be either 'cmd_vel' or 'environment'")
    if ood_feature == 'environment':
        if not environment_windows or not all(
            isinstance(window, (list, tuple)) and len(window) == 2 and window[0] <= window[1]
            for window in environment_windows
        ):
            raise ValueError(
                'environment_windows must be a non-empty list of [min, max] windows with min <= max'
            )


def csv_matches_environment_windows(csv_path, environment_windows):
    """Return ``(matches, environment_id)`` for an environment-selected CSV."""
    environment_id = environment_number_from_filename(csv_path)
    if environment_id is None:
        raise ValueError(
            f'Cannot determine environment number from CSV filename: {csv_path}. '
            'Use a name ending in _<number>.csv or containing env<number>.'
        )
    return any(low <= environment_id <= high for low, high in environment_windows), environment_id
