import os
import comet_ml
import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score
from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.feature_encoder import get_encoder_from_constraints
from src.attacks.moeva2.moeva2 import Moeva2
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.attacks.moeva2.utils import results_to_numpy_results
from src.config_parser.config_parser import get_config
from src.experiments.botnet.features import augment_data
from src.experiments.botnet.model import train_model, print_score
from src.experiments.united.utils import get_constraints_from_str
from src.utils.comet import init_comet
from src.utils.in_out import load_model
from src.attacks.pgd.classifier import TF2Classifier as kc
from src.attacks.pgd.atk import PGDTF2 as PGD

np.random.seed(205)
import tensorflow as tf

tf.random.set_seed(206)

from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# ----- CONFIG
config = get_config()
project_name = config["project_name"]
nb_important_features = 19
threshold = config["misclassification_threshold"]

# ----- LOAD

x_train = np.load(f"./data/{project_name}/X_train.npy")
x_test = np.load(f"./data/{project_name}/X_test.npy")
y_train = np.load(f"./data/{project_name}/y_train.npy")
y_test = np.load(f"./data/{project_name}/y_test.npy")
features = pd.read_csv(f"./data/{project_name}/features.csv")
constraints = pd.read_csv(f"./data/{project_name}/constraints.csv")

# ----- SCALER

scaler_path = f"./models/{project_name}/scaler.joblib"
if os.path.exists(scaler_path):
    print(f"{scaler_path} exists loading...")
    scaler = joblib.load(scaler_path)
else:
    scaler = MinMaxScaler()
    x_all = np.concatenate((x_train, x_test))
    x_min = features["min"]
    x_max = features["max"]
    x_min[x_min == "dynamic"] = np.min(x_all, axis=0)[x_min == "dynamic"]
    x_max[x_max == "dynamic"] = np.max(x_all, axis=0)[x_max == "dynamic"]
    x_min = x_min.astype(np.float).to_numpy().reshape(1, -1)
    x_max = x_max.astype(np.float).to_numpy().reshape(1, -1)
    x_min = np.min(np.concatenate((x_min, x_all)), axis=0).reshape(1, -1)
    x_max = np.max(np.concatenate((x_max, x_all)), axis=0).reshape(1, -1)
    scaler.fit(np.concatenate((np.floor(x_min), np.ceil(x_max))))
    joblib.dump(scaler, scaler_path)

# ----- TRAIN MODEL

model_path = f"./models/{project_name}/nn.model"

if os.path.exists(model_path):
    print(f"{model_path} exists loading...")
    model = load_model(model_path)
else:
    model = train_model(scaler.transform(x_train), to_categorical(y_train))
    tf.keras.models.save_model(
        model,
        model_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )
# ----- MODEL SCORE

y_proba = model.predict_proba(scaler.transform(x_test))
y_pred = (y_proba[:, 1] >= threshold).astype(int)

# ----- FIND IMPORTANT FEATURES

important_features_path = f"./data/{project_name}/important_features_19.npy"
if os.path.exists(important_features_path):
    print(f"{important_features_path} exists loading...")
    important_features = np.load(important_features_path)
else:
    sampler = RandomUnderSampler(sampling_strategy={0: 300, 1: 300}, random_state=42)
    x_train_small, y_train_small = sampler.fit_resample(x_train, y_train)
    explainer = shap.DeepExplainer(model, scaler.transform(x_train_small))
    shap_values = explainer.shap_values(scaler.transform(x_train_small))
    shap_values_per_feature = np.mean(np.abs(np.array(shap_values)[0]), axis=0)
    shap_values_per_mutable_feature = shap_values_per_feature[features["mutable"]]

    mutable_feature_index = np.where(features["mutable"])[0]
    order_feature_mutable = np.argsort(shap_values_per_mutable_feature)[::-1]
    important_features_index = mutable_feature_index[order_feature_mutable][
        :nb_important_features
    ]
    important_features_mean = np.mean(x_train[:, important_features_index], axis=0)

    important_features = np.column_stack(
        [important_features_index, important_features_mean]
    )
    np.save(important_features_path, important_features)


# ----- AUGMENT DATASET
x_train_augmented_path = f"./data/{project_name}/x_train_augmented_19.npy"
x_test_augmented_path = f"./data/{project_name}/x_test_augmented_19.npy"
features_augmented_path = f"./data/{project_name}/features_augmented_19.csv"
constraints_augmented_path = f"./data/{project_name}/constraints_augmented_19.csv"
if os.path.exists(x_train_augmented_path) and os.path.exists(x_test_augmented_path):
    x_train_augmented = np.load(x_train_augmented_path)
    x_test_augmented = np.load(x_test_augmented_path)
    features_augmented = pd.read_csv(features_augmented_path)
    constraints_augmented = pd.read_csv(constraints_augmented_path)
    nb_new_features = x_train_augmented.shape[1] - x_train.shape[1]
else:
    x_train_augmented = augment_data(x_train, important_features)
    x_test_augmented = augment_data(x_test, important_features)
    nb_new_features = x_train_augmented.shape[1] - x_train.shape[1]
    features_augmented = features.append(
        [
            {
                "feature": f"augmented_{i}",
                "type": "int",
                "mutable": True,
                "min": 0.0,
                "max": 1.0,
                "augmentation": True,
            }
            for i in range(nb_new_features)
        ]
    )
    constraints_augmented = constraints.append(
        [
            {
                "min": 0.0,
                "max": 1.0,
                "augmentation": True,
            }
            for i in range(nb_new_features)
        ]
    )
    np.save(x_train_augmented_path, x_train_augmented)
    np.save(x_test_augmented_path, x_test_augmented)
    features_augmented.to_csv(features_augmented_path)
    constraints_augmented.to_csv(constraints_augmented_path)

# ----- Augmented scaler

scaler_augmented_path = f"./models/{project_name}/scaler_augmented_19.joblib"

if os.path.exists(scaler_augmented_path):
    scaler_augmented = joblib.load(scaler_augmented_path)
else:
    scaler_augmented = MinMaxScaler()
    scaling_data = np.concatenate(
        (
            np.concatenate((scaler.data_min_, np.zeros(nb_new_features))),
            np.concatenate((scaler.data_max_, np.ones(nb_new_features))),
        ),
        axis=0,
    ).reshape(2, -1)

    scaler_augmented.fit(scaling_data)
    joblib.dump(scaler_augmented, scaler_augmented_path)

# ----- TRAIN MODEL

model_augmented_path = f"./models/{project_name}/nn_augmented_19.model"

if os.path.exists(model_augmented_path):
    print(f"{model_augmented_path} exists loading...")
    model_augmented = load_model(model_augmented_path)
else:
    model_augmented = train_model(
        scaler_augmented.transform(x_train_augmented), to_categorical(y_train)
    )
    tf.keras.models.save_model(
        model_augmented,
        model_augmented_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )
# ----- MODEL SCORE

y_proba = model_augmented.predict_proba(scaler_augmented.transform(x_test_augmented))
y_pred_augmented = (y_proba[:, 1] >= threshold).astype(int)
print(f"AUROC: {roc_auc_score(y_test, y_proba[:, 1])}")


# ----- ADVERSARIAL TRAINING MOEVA

x_train_adv_moeva_path = f"./data/{project_name}/x_train_adv_moeva.npy"

if os.path.exists(x_train_adv_moeva_path):
    x_train_adv_moeva = np.load(x_train_adv_moeva_path)
else:
    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    x_train_candidates = x_train[
        (y_train == 1)
        * (
            y_train
            == (
                model.predict_proba(scaler.transform(x_train))[:, 1] >= threshold
            ).astype(int)
        )
    ]
    print(f"{x_train_candidates.shape} candidates.")
    constraints.check_constraints_error(x_train_candidates)
    n_gen = config["budget"]
    objective_calc = ObjectiveCalculator(
        Classifier(model),
        constraints,
        minimize_class=1,
        thresholds={"f1": threshold, "f2": config["eps"]},
        min_max_scaler=scaler,
        ml_scaler=scaler,
        norm=config["norm"],
    )
    moeva = Moeva2(
        model_path,
        constraints,
        problem_class=None,
        l2_ball_size=0.0,
        norm=config["norm"],
        n_gen=n_gen,
        n_pop=config["n_pop"],
        n_offsprings=config["n_offsprings"],
        scale_objectives=False,
        save_history=config.get("save_history"),
        seed=config["seed"],
        n_jobs=config["system"]["n_jobs"],
        ml_scaler=scaler,
        verbose=1,
    )
    x_train_attacks = results_to_numpy_results(
        moeva.generate(x_train_candidates, 1), get_encoder_from_constraints(constraints)
    )
    x_train_adv_moeva = objective_calc.get_successful_attacks(
        x_train_candidates,
        x_train_attacks,
        preferred_metrics="misclassification",
        order="asc",
        max_inputs=1,
    )
    print(f"Success rate: {x_train_adv_moeva.shape[0] / x_train_attacks.shape[0]}")
    print(f"Retraining with: {x_train_adv_moeva.shape[0]}")

    np.save(x_train_adv_moeva_path, x_train_adv_moeva)

model_adv_moeva_path = f"./models/{project_name}/nn_moeva.model"
if os.path.exists(model_adv_moeva_path):
    model_adv_moeva = load_model(model_adv_moeva_path)
else:
    x_train_local = np.concatenate((x_train, x_train_adv_moeva), axis=0)
    y_train_local = np.concatenate(
        (y_train, np.ones(x_train_adv_moeva.shape[0])), axis=0
    )

    model_adv_moeva = train_model(
        scaler.transform(x_train_local), to_categorical(y_train_local)
    )
    tf.keras.models.save_model(
        model_adv_moeva,
        model_adv_moeva_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )

y_proba = model_adv_moeva.predict_proba(scaler.transform(x_test))
y_pred_adv_moeva = (y_proba[:, 1] >= threshold).astype(int)
print(f"AUROC: {roc_auc_score(y_test, y_proba[:, 1])}")


# ----- ADVERSARIAL TRAINING GRADIENT

x_train_adv_gradient_path = f"./data/{project_name}/x_train_adv_gradient.npy"

if os.path.exists(x_train_adv_gradient_path):
    x_train_adv_gradient = np.load(x_train_adv_gradient_path)
else:
    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    experiment = None
    enable_comet = config.get("comet", True)
    if enable_comet:
        params = config
        experiment = init_comet(params)

    x_train_candidates = x_train[
        (y_train == 1)
        * (
                y_train
                == (
                        model.predict_proba(scaler.transform(x_train))[:, 1] >= threshold
                ).astype(int)
        )
        ]
    print(f"{x_train_candidates.shape} candidates.")
    constraints.check_constraints_error(x_train_candidates)

    objective_calc = ObjectiveCalculator(
        Classifier(model),
        constraints,
        minimize_class=1,
        thresholds={"f1": threshold, "f2": config["eps"]},
        min_max_scaler=scaler,
        ml_scaler=scaler,
        norm=config["norm"],
    )

    initial_shape = x_train.shape[1:]
    new_input = tf.keras.layers.Input(shape=initial_shape)
    model_att = tf.keras.models.Model(inputs=[new_input], outputs=[model(new_input)])
    kc_classifier = kc(
        model_att,
        clip_values=(0.0, 1.0),
        input_shape=initial_shape,
        loss_object=tf.keras.losses.categorical_crossentropy,
        nb_classes=2,
        constraints=constraints,
        scaler=scaler,
        experiment=experiment,
        parameters=config,
    )
    attack = PGD(
        kc_classifier,
        eps=config["eps"] - 0.000001,
        eps_step=0.1,
        norm=config.get("norm"),
        verbose=config["system"]["verbose"] == 1,
        max_iter=int(config.get("budget")),
        num_random_init=config.get("nb_random", 0),
        batch_size=x_train_candidates.shape[0],
        loss_evaluation=config.get("loss_evaluation"),
    )

    x_train_attacks = scaler.inverse_transform(
        attack.generate(
            x=scaler.transform(x_train_candidates),
            y=np.zeros(x_train_candidates.shape[0]),
            mask=constraints.get_mutable_mask(),
        )
    )
    mask_int = constraints.get_feature_type() != "real"
    x_plus_minus = x_train_attacks[:, mask_int] - x_train_candidates[:, mask_int] >= 0
    x_train_attacks[:, mask_int][x_plus_minus] = np.floor(
        x_train_attacks[:, mask_int][x_plus_minus]
    )
    x_train_attacks[:, mask_int][~x_plus_minus] = np.ceil(
        x_train_attacks[:, mask_int][~x_plus_minus]
    )

    if len(x_train_attacks.shape) == 2:
        x_train_attacks = x_train_attacks[:, np.newaxis, :]
    np.save("./tmp.npy", x_train_attacks)
    x_train_adv_gradient = objective_calc.get_successful_attacks(
        x_train_candidates,
        x_train_attacks,
        preferred_metrics="misclassification",
        order="asc",
        max_inputs=1,
    )

    print(f"Success rate: {x_train_adv_gradient.shape[0] / x_train_attacks.shape[0]}")
    print(f"Retraining with: {x_train_adv_gradient.shape[0]}")
    np.save(x_train_adv_gradient_path, x_train_adv_gradient)

# ----- Common x_attacks
x_candidates_path = f"./data/{project_name}/x_candidates_common.npy"
x_candidates_augmented_path = f"./data/{project_name}/x_candidates_common_augmented.npy"

if os.path.exists(x_candidates_path) and os.path.exists(x_candidates_augmented_path):
    x_candidates = np.load(x_candidates_path)
    x_candidates_augmented = np.load(x_candidates_augmented_path)
else:
    candidates_index = (y_test == 1) * (y_test == y_pred) * (y_test == y_pred_augmented) * (y_test == y_pred_adv_moeva)
    x_candidates = x_test[candidates_index, :]
    x_candidates_augmented = x_test_augmented[candidates_index, :]
    np.save(x_candidates_path, x_candidates)
    np.save(x_candidates_augmented_path, x_candidates_augmented)

print(f"Candidates: {x_candidates.shape}.")
