from itertools import combinations
from typing import Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.attacks.moeva2.constraints import Constraints
import autograd.numpy as anp
import pandas as pd
import tensorflow as tf
import logging

from src.attacks.moeva2.utils import get_ohe_masks
from src.examples.utils import constraints_augmented_np, constraints_augmented_tf
from src.experiments.botnet.features import augment_data


class LcldAugmentedConstraints(Constraints):
    def __init__(
        self,
        feature_path: str,
        constraints_path: str,
        import_features_path = None,
    ):
        self._provision_constraints_min_max(constraints_path)
        self._provision_feature_constraints(feature_path)
        self._fit_scaler()
        if import_features_path is None:
            import_features_path = "./data/lcld/important_features.npy"
        self.important_features = np.load(import_features_path)

    def _fit_scaler(self) -> None:
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        min_c, max_c = self.get_constraints_min_max()
        self._scaler = self._scaler.fit([min_c, max_c])

    @staticmethod
    def _date_feature_to_month(feature):
        return np.floor(feature / 100) * 12 + (feature % 100)

    @staticmethod
    def _date_feature_to_month_tf(feature):
        return tf.math.floor(feature / 100) * 12 + tf.math.floormod(feature, 100)

    def fix_features_types(self, x):

        new_tensor_v = tf.Variable(x)

        # enforcing 2 possibles values
        x1 = tf.where(x[:, 1] < (60 + 36) / 2, 36 * tf.ones_like(x[:, 1]), 60)
        new_tensor_v = new_tensor_v[:, 1].assign(x1)

        x0 = x[:, 0]
        x2 = x[:, 2] / 1200

        # enforcing the power formula
        x3 = x0 * x2 * tf.math.pow(1 + x2, x1) / (tf.math.pow(1 + x2, x1) - 1)
        new_tensor_v = new_tensor_v[:, 3].assign(x3)

        # enforcing ohe

        ohe_masks = get_ohe_masks(self._feature_type)

        new_tensor_v = new_tensor_v.numpy()
        for mask in ohe_masks:
            ohe = new_tensor_v[:, mask]
            max_feature = np.argmax(ohe, axis=1)
            new_ohe = np.zeros_like(ohe)
            new_ohe[np.arange(len(ohe)), max_feature] = 1

            new_tensor_v[:, mask] = new_ohe

        if new_tensor_v.shape[1] > 47:
            combi = -sum(1 for i in combinations(range(len(self.important_features)), 2))
            new_tensor_v = new_tensor_v[..., :combi]
            new_tensor_v = augment_data(new_tensor_v, self.important_features)

        return tf.convert_to_tensor(new_tensor_v)

    def evaluate_tf2(self, x):
        # ----- PARAMETERS

        tol = 1e-3
        alpha = 1e-5

        # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2] / 1200
        x3 = x[:, 3]

        calculated_installment = (
            x0 * x2 * tf.math.pow(1 + x2, x1) / (tf.math.pow(1 + x2, x1) - 1)
        )

        calculated_installment_36 = (
            x0 * x2 * tf.math.pow(1 + x2, 36) / (tf.math.pow(1 + x2, 36) - 1)
        )

        calculated_installment_60 = (
            x0 * x2 * tf.math.pow(1 + x2, 60) / (tf.math.pow(1 + x2, 60) - 1)
        )

        g41 = (
            tf.minimum(
                tf.math.abs(x3 - calculated_installment_36),
                tf.math.abs(x3 - calculated_installment_60),
            )
            - 0.099999
        )
        g41_ = tf.math.abs(x3 - calculated_installment) - 0.099999

        # open_acc <= total_acc
        g42 = alpha + x[:, 10] - x[:, 14]
        g42 = tf.clip_by_value(g42, 0, tf.constant(np.inf))

        # pub_rec_bankruptcies <= pub_rec
        g43 = tf.clip_by_value(alpha + x[:, 16] - x[:, 11], 0, tf.constant(np.inf))

        # term = 36 or term = 60
        g44 = tf.math.minimum(tf.math.abs(36 - x[:, 1]), tf.math.abs(60 - x[:, 1]))

        # ratio_loan_amnt_annual_inc
        g45 = tf.math.abs(x[:, 20] - x[:, 0] / x[:, 6])

        # ratio_open_acc_total_acc
        g46 = tf.math.abs(x[:, 21] - x[:, 10] / x[:, 14])

        # diff_issue_d_earliest_cr_line
        g47 = tf.math.abs(
            x[:, 22]
            - (
                self._date_feature_to_month_tf(x[:, 7])
                - self._date_feature_to_month_tf(x[:, 9])
            )
        )

        # ratio_pub_rec_diff_issue_d_earliest_cr_line
        g48 = tf.math.abs(x[:, 23] - x[:, 11] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        g49 = tf.math.abs(x[:, 24] - x[:, 16] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        # ratio_mask = x[:, 11] == 0
        # ratio = torch.empty(x.shape[0])
        # ratio = np.ma.masked_array(ratio, mask=ratio_mask, fill_value=-1).filled()
        # ratio[~ratio_mask] = x[~ratio_mask, 16] / x[~ratio_mask, 11]
        # ratio[ratio == np.inf] = -1
        # ratio[np.isnan(ratio)] = -1

        broken = x[:, 16] / x[:, 11]
        ratio = x[:, 16] / (x[:, 11] + alpha)
        # g410 = tf.math.abs(x[:, 25] - x[:, 16] / (x[:, 11] + alpha))
        clean_ratio = tf.where(tf.math.is_nan(broken), -1 * tf.ones_like(ratio), ratio)
        g410 = tf.math.abs(x[:, 25] - clean_ratio)

        constraints = tf.stack(
            [g41, g42, g43, g44, g45, g46, g47, g48, g49, g410],
            # + constraints_augmented_tf(x, self.important_features[:, 0], self.important_features[:, 1]),
            1,
        )

        constraints = tf.clip_by_value(constraints - tol, 0, tf.constant(np.inf))

        return constraints
        # return tf.nn.softmax(constraints) * tf.reduce_max(constraints)
        # print(max_constraints.cpu().detach())
        # return max_constraints.mean()

    def evaluate(self, x: np.ndarray, use_tensors: bool = False) -> np.ndarray:
        if use_tensors:
            return self.evaluate_tf2(x)
        else:
            return self.evaluate_numpy(x)

    def evaluate_numpy(self, x: np.ndarray) -> np.ndarray:
        # ----- PARAMETERS

        tol = 1e-3

        # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
        calculated_installment = (
            x[:, 0] * (x[:, 2] / 1200) * (1 + x[:, 2] / 1200) ** x[:, 1]
        ) / ((1 + x[:, 2] / 1200) ** x[:, 1] - 1)
        g41 = np.absolute(x[:, 3] - calculated_installment) - 0.099999

        # open_acc <= total_acc
        g42 = x[:, 10] - x[:, 14]

        # pub_rec_bankruptcies <= pub_rec
        g43 = x[:, 16] - x[:, 11]

        # term = 36 or term = 60
        g44 = np.absolute((36 - x[:, 1]) * (60 - x[:, 1]))

        # ratio_loan_amnt_annual_inc
        g45 = np.absolute(x[:, 20] - x[:, 0] / x[:, 6])

        # ratio_open_acc_total_acc
        g46 = np.absolute(x[:, 21] - x[:, 10] / x[:, 14])

        # diff_issue_d_earliest_cr_line
        g47 = np.absolute(
            x[:, 22]
            - (
                self._date_feature_to_month(x[:, 7])
                - self._date_feature_to_month(x[:, 9])
            )
        )

        # ratio_pub_rec_diff_issue_d_earliest_cr_line
        g48 = np.absolute(x[:, 23] - x[:, 11] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        g49 = np.absolute(x[:, 24] - x[:, 16] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        ratio_mask = x[:, 11] == 0
        ratio = np.empty(x.shape[0])
        ratio = np.ma.masked_array(ratio, mask=ratio_mask, fill_value=-1).filled()
        ratio[~ratio_mask] = x[~ratio_mask, 16] / x[~ratio_mask, 11]
        ratio[ratio == np.inf] = -1
        ratio[np.isnan(ratio)] = -1
        g410 = np.absolute(x[:, 25] - ratio)

        # Augmented

        constraints = anp.column_stack(
            [g41, g42, g43, g44, g45, g46, g47, g48, g49, g410]
            + constraints_augmented_np(x, self.important_features[:, 0], self.important_features[:, 1])
        )
        constraints[constraints <= tol] = 0.0

        return constraints

    def get_nb_constraints(self) -> int:
        return 10 + 10

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return self._scaler.transform(x)

    def get_constraints_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._constraints_min, self._constraints_max

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask

    def get_feature_min_max(self, dynamic_input=None) -> Tuple[np.ndarray, np.ndarray]:

        # By default min and max are the extreme values
        feature_min = np.array([0.0] * self._feature_min.shape[0])
        feature_max = np.array([0.0] * self._feature_max.shape[0])

        # Creating the mask of value that should be provided by input
        min_dynamic = self._feature_min.astype(str) == "dynamic"
        max_dynamic = self._feature_max.astype(str) == "dynamic"

        # Replace de non dynamic value by the value provided in the definition
        feature_min[~min_dynamic] = self._feature_min[~min_dynamic]
        feature_max[~max_dynamic] = self._feature_max[~max_dynamic]

        # If the dynamic input was provided, replace value for output, else do nothing (keep the extreme values)
        if dynamic_input is not None:
            feature_min[min_dynamic] = dynamic_input[min_dynamic]
            feature_max[max_dynamic] = dynamic_input[max_dynamic]

        # Raise warning if dynamic input waited but not provided
        dynamic_number = min_dynamic.sum() + max_dynamic.sum()
        if dynamic_number > 0 and dynamic_input is None:
            logging.getLogger().warning(
                f"{dynamic_number} feature min and max are dynamic but no input were provided."
            )

        return feature_min, feature_max

    def get_feature_type(self) -> np.ndarray:
        return self._feature_type

    def _provision_feature_constraints(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._feature_min = df["min"].to_numpy()
        self._feature_max = df["max"].to_numpy()
        self._mutable_mask = df["mutable"].to_numpy()
        self._feature_type = df["type"].to_numpy()

    def _provision_constraints_min_max(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._constraints_min = df["min"].to_numpy()
        self._constraints_max = df["max"].to_numpy()
        self._fit_scaler()
