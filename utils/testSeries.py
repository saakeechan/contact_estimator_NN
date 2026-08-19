"""Run one configured single-test evaluator over a seed range."""
from argparse import Namespace
import sys

from base_test_series import BaseSeriesTest


# Edit these values, then run: python3 utils/testSeries.py
MODEL = 'natpn'  # 'natpn', 'der'
FIRST_SEED = 3101
LAST_SEED = 3300
OOD_FEATURE = 'cmd_vel'  # 'cmd_vel' or 'environment'
CMD_VEL_X_WINDOW = (0.0, 3.0)
ENVIRONMENT_WINDOWS = [(0, 20)]  # Inclusive; add tuples for disjoint windows.
PLOT_METRIC = 'negative-log-density'  # 'epistemic' or 'negative-log-density'
RUN_KNN = False  # Set False to skip kNN/OOD evaluation for every seed.

MODEL_SCRIPTS = {
    'natpn': 'src/testSingleNatPN.py',
    'der': 'src/testSingleDER.py',
}


class SeriesTest(BaseSeriesTest):
    description = 'Run the configured model over a seed range and collect its metrics in a CSV.'
    single_test_script = MODEL_SCRIPTS[MODEL]
    output_prefix = MODEL
    default_ood_feature = OOD_FEATURE
    default_cmd_vel_x_window = CMD_VEL_X_WINDOW
    default_environment_windows = ENVIRONMENT_WINDOWS
    default_plot_metric = PLOT_METRIC
    run_knn = RUN_KNN
    first_seed = FIRST_SEED
    last_seed = LAST_SEED

    def create_in_process_runner(self, args, root, knn_cache_path):
        sys.path.insert(0, str(root / 'src'))
        if MODEL == 'natpn':
            from testSingleNatPN import NatPNSingleTest, run_evaluation
        else:
            from testSingleDER import DERSingleTest, run_evaluation

        evaluator = NatPNSingleTest() if MODEL == 'natpn' else DERSingleTest()
        model = checkpoint_path = None

        def run_seed(seed):
            nonlocal model, checkpoint_path
            single_args = Namespace(
                config_name=str(args.config_name), seed=seed,
                cmd_vel_x_window=args.cmd_vel_x_window, ood_feature=args.ood_feature,
                environment_window=args.environment_window, metrics_json=None,
                skip_umap=not args.with_umap, save_umap=args.with_umap,
                skip_knn=args.skip_knn, knn_cache=str(knn_cache_path), skip_plots=True,
            )
            context = evaluator.prepare(single_args)
            metrics, model, checkpoint_path = run_evaluation(
                single_args, context, model=model, checkpoint_path=checkpoint_path
            )
            return metrics

        return run_seed


if __name__ == '__main__':
    SeriesTest().main()
