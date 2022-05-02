import os
import time
from itertools import combinations
from pathlib import Path
import comet_ml
import joblib
import numpy as np
import tensorflow as tf

from src.attacks.pgd.atk import PGDTF2 as PGD
from src.attacks.pgd.auto_pgd import AutoProjectedGradientDescent
from src.attacks.pgd.classifier import TF2Classifier as kc
from src.attacks.sat.sat import SatAttack
from src.config_parser.config_parser import get_config, get_config_hash, save_config
from src.experiments.botnet.features import augment_data
from src.experiments.united.utils import (
    get_constraints_from_str,
    get_sat_constraints_from_str,
)
from src.utils import in_out, filter_initial_states
from src.utils.comet import init_comet
from src.utils.in_out import load_model


from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator


def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    out_dir = config["dirs"]["results"]
    config_hash = get_config_hash()
    mid_fix = f"{config['attack_name']}_{config['loss_evaluation']}"
    metrics_path = f"{out_dir}/metrics_{mid_fix}_{config_hash}.json"
    if os.path.exists(metrics_path):
        print(f"Configuration with hash {config_hash} already executed. Skipping")
        exit(0)

    tf.random.set_seed(config["seed"])
    experiment = None
    enable_comet = config.get("comet", True)
    if enable_comet:
        params = config
        experiment = init_comet(params)

    apply_sat = "sat" in config["loss_evaluation"]

    Path(config["dirs"]["results"]).mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    if config["paths"].get("important_features", False):
        constraints = get_constraints_from_str(config["project_name"])(
            config["paths"]["features"],
            config["paths"]["constraints"],
            config["paths"].get("important_features")
        )
    else:
        constraints = get_constraints_from_str(config["project_name"])(
            config["paths"]["features"],
            config["paths"]["constraints"],
        )

    x_initial = np.load(config["paths"]["x_candidates"])

    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )

    initial_shape = x_initial.shape[1:]

    model_base = load_model(config["paths"]["model"])
    scaler = joblib.load(config["paths"]["ml_scaler"])

    # ----- Check constraints

    constraints.check_constraints_error(x_initial)

    # ----- Perform the attack

    start_time = time.time()

    new_input = tf.keras.layers.Input(shape=initial_shape)
    model = tf.keras.models.Model(inputs=[new_input], outputs=[model_base(new_input)])
    kc_classifier = kc(
        model,
        clip_values=(0.0, 1.0),
        input_shape=initial_shape,
        loss_object=tf.keras.losses.categorical_crossentropy,
        nb_classes=2,
        constraints=constraints,
        scaler=scaler,
        experiment=experiment,
        parameters=config,
    )
    # Use only half eps if apply sat after
    per_attack_eps = config["eps"] / 2 if apply_sat else config["eps"]
    if "autopgd" in config.get("loss_evaluation"):
        attack = AutoProjectedGradientDescent(
            kc_classifier,
            constraints=constraints,
            scaler=scaler,
            experiment=experiment,
            parameters=config,
            eps=per_attack_eps - 0.000001,
            eps_step=per_attack_eps / 3,
            loss_type="cross_entropy",
            nb_random_init=config.get("nb_random", 1),
            max_iter=int(config.get("budget")),
            batch_size=x_initial.shape[0],
        )
    else:
        attack = PGD(
            kc_classifier,
            eps=per_attack_eps - 0.000001,
            eps_step=0.1,
            norm=config.get("norm"),
            verbose=config["system"]["verbose"] == 1,
            max_iter=int(config.get("budget")),
            num_random_init=config.get("nb_random", 0),
            batch_size=x_initial.shape[0],
            loss_evaluation=config.get("loss_evaluation"),
        )
    x_attacks = scaler.inverse_transform(
        attack.generate(
            x=scaler.transform(x_initial),
            mask=constraints.get_mutable_mask(),
        )
    )
    mask_int = constraints.get_feature_type() != "real"
    x_plus_minus = x_attacks[:, mask_int] - x_initial[:, mask_int] >= 0
    x_attacks[:, mask_int][x_plus_minus] = np.floor(
        x_attacks[:, mask_int][x_plus_minus]
    )
    x_attacks[:, mask_int][~x_plus_minus] = np.ceil(
        x_attacks[:, mask_int][~x_plus_minus]
    )
    # x_attacks[:, mask_int] = np.rint(x_attacks[:, mask_int])

    # Apply sat if needed

    if apply_sat:
        sat_constraints = get_sat_constraints_from_str(config["project_name"])
        attack = SatAttack(
            constraints,
            sat_constraints,
            scaler,
            per_attack_eps,
            np.inf,
            n_sample=1,
            verbose=1,
            n_jobs=config["system"]["n_jobs"],
        )
        x_attacks = attack.generate(x_initial, x_attacks)

    if config["reconstruction"]:
        important_features = np.load("./data/lcld/important_features.npy")
        combi = -sum(1 for i in combinations(range(len(important_features)), 2))
        x_attacks_l = x_attacks[:, :combi]
        print(x_attacks_l.shape)
        x_attacks = augment_data(x_attacks_l, important_features)
        print(x_attacks.shape)

    consumed_time = time.time() - start_time
    # ----- End attack

    if len(x_attacks.shape) == 2:
        x_attacks = x_attacks[:, np.newaxis, :]
    classifier = Classifier(model_base)
    threholds = {"f1": config["misclassification_threshold"], "f2": config["eps"]}
    objective_calc = ObjectiveCalculator(
        classifier,
        constraints,
        minimize_class=1,
        thresholds=threholds,
        min_max_scaler=scaler,
        ml_scaler=scaler,
        norm=config["norm"],
    )

    success_rate_df = objective_calc.success_rate_3d_df(x_initial, x_attacks)
    print(success_rate_df)

    if config["comet"]:
        for c, v in zip(success_rate_df.columns, success_rate_df.values[0]):
            experiment.log_metric(c, v)

    # Save
    # X_attacks

    x_attacks_path = f"{out_dir}/x_attacks_{mid_fix}_{config_hash}.npy"
    np.save(x_attacks_path, x_attacks)
    # experiment.log_asset(x_attacks_path)

    # History
    if config.get("save_history") in ["reduced", "full"]:
        history = np.swapaxes(np.array(kc_classifier.history), 0, 1)
        history = history[:, :, np.newaxis, :]
        np.save(f"{out_dir}/x_history_{config_hash}.npy", history)

    # Metrics
    metrics = {
        "objectives": success_rate_df.to_dict(orient="records")[0],
        "time": consumed_time,
        "config": config,
        "config_hash": config_hash,
    }
    success_rate_df.to_csv(
        f"{out_dir}/success_rate_{mid_fix}_{config_hash}.csv", index=False
    )

    in_out.json_to_file(metrics, metrics_path)

    # Config
    save_config(f"{out_dir}/config_{mid_fix}_")


if __name__ == "__main__":
    config = get_config()
    run()
    # To allow the metrics to be uploaded
    # time.sleep(30)
