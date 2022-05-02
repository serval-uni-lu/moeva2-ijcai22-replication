from data.config import COMET_APIKEY
from comet_ml import Experiment
import time


def init_comet(args, project_name="moeva", experiment_name=None):
    timestamp = time.time()
    args["timestamp"] = timestamp
    if experiment_name is None:
        experiment_name = "{}_{}".format(args["attack_name"], timestamp)
    else:
        experiment_name = "{}_{}".format(experiment_name, timestamp)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=project_name,
        auto_param_logging=False,
        auto_metric_logging=False,
        parse_args=False,
        display_summary_level=None,
        disabled=False,
    )

    experiment.set_name(experiment_name)
    experiment.log_parameters(args)

    return experiment
