import argparse
import os

import yaml

try:
    from utils.csv2numpyIsaac import csv2numpy_split, resolve_active_legs
except ModuleNotFoundError:  # Supports `python utils/csv2numpyMujoco.py`.
    from csv2numpyIsaac import csv2numpy_split, resolve_active_legs


# Edit these when converting a different MuJoCo CSV dataset.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FOLDER = os.path.join(PROJECT_ROOT, 'Data/MujocoCSVFiles/Slope/')
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'Data/MujocoNumpyFiles/Slope/')


# Keep MuJoCo's source-field contract here; conversion flow lives in the shared
# Isaac converter because both simulators produce the same canonical arrays.
MUJOCO_SCHEMA = {
    'time': 'time',
    'command_velocity': 'command_twist_linear_x',
    'imu_acceleration': ('lowstate_accel_x', 'lowstate_accel_y', 'lowstate_accel_z'),
    'imu_angular_rate': ('lowstate_gyro_x', 'lowstate_gyro_y', 'lowstate_gyro_z'),
    'joint_position_pattern': '{joint}_q',
    'joint_velocity_pattern': '{joint}_qd',
    'joint_torque_pattern': '{joint}_tau_est',
    'foot_position_pattern': 'fk_{leg}_foot_pos_{axis}',
    'foot_velocity_pattern': 'fk_{leg}_foot_vel_{axis}',
    'contact_columns': {'left': 'sport_foot_force_0', 'right': 'sport_foot_force_1'},
    'contacts_positive': True,
    'world_velocity': ('sport_velocity_world_x', 'sport_velocity_world_y', 'sport_velocity_world_z'),
    'quaternion': ('lowstate_quat_w', 'lowstate_quat_x', 'lowstate_quat_y', 'lowstate_quat_z'),
}


def csv2numpy_mujoco_split(data_pth, save_pth, cmd_vel_x_windows=((0.0, 2.0),),
                           ood_feature='cmd_vel', environment_windows=((0, 0),), legs=('left', 'right')):
    """Convert MuJoCo CSVs using the shared canonical-array conversion flow."""
    return csv2numpy_split(
        data_pth,
        save_pth,
        cmd_vel_x_windows=cmd_vel_x_windows,
        ood_feature=ood_feature,
        environment_windows=environment_windows,
        legs=legs,
        schema=MUJOCO_SCHEMA,
    )


def main():
    parser = argparse.ArgumentParser(description='Convert MuJoCo CSV to numpy.')
    parser.add_argument(
        '--config_name',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'config/network_params.yaml'),
    )
    args = parser.parse_args()

    with open(args.config_name) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    ood_feature = config.get('ood_feature', 'cmd_vel')
    legs = resolve_active_legs(config.get('active_legs', 'both'))
    if ood_feature not in ('cmd_vel', 'environment'):
        raise ValueError("ood_feature must be either 'cmd_vel' or 'environment'")
    window_key = 'cmd_vel_x_windows' if ood_feature == 'cmd_vel' else 'environment_windows'
    windows = config.get(window_key, [[0.0, 2.0]] if ood_feature == 'cmd_vel' else [[0, 10]])
    if not windows or not all(
        isinstance(window, (list, tuple)) and len(window) == 2 and window[0] <= window[1]
        for window in windows
    ):
        raise ValueError(f'{window_key} must be a non-empty list of [min, max] windows with min <= max')

    print('Using configuration:')
    print(f'  CSV folder: {CSV_FOLDER}')
    print(f'  Save path: {DATA_FOLDER}')
    print(f'  OOD feature: {ood_feature}')
    print(f"  Active legs: {', '.join(legs)}")
    print(f"  {'Command-velocity' if ood_feature == 'cmd_vel' else 'Environment'} windows: {windows}")

    csv2numpy_mujoco_split(
        CSV_FOLDER,
        DATA_FOLDER,
        cmd_vel_x_windows=windows if ood_feature == 'cmd_vel' else (),
        ood_feature=ood_feature,
        environment_windows=windows if ood_feature == 'environment' else (),
        legs=legs,
    )


if __name__ == '__main__':
    main()
