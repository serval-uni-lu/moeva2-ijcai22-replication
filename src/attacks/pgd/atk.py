from art.attacks.evasion import ProjectedGradientDescentTensorFlowV2
from typing import Optional, Union, TYPE_CHECKING
from art.utils import compute_success, random_sphere, compute_success_array
import numpy as np
from art.config import ART_NUMPY_DTYPE


import logging

logger = logging.getLogger(__name__)


class PGDTF2(ProjectedGradientDescentTensorFlowV2):
    def __init__(
        self,
        estimator: "TensorFlowV2Classifier",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        tensor_board: Union[str, bool] = False,
        verbose: bool = True,
        loss_evaluation: str = ""
    ):
        self.iter_counter = 0
        self.loss_evaluation = loss_evaluation
        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            tensor_board=tensor_board,
            verbose=verbose,
        )

    def _compute_multi_objectives(self, x: "tf.Tensor", x_init: "tf.Tensor"):

        from src.utils.in_out import load_model
        from src.attacks.moeva2.classifier import Classifier
        from src.attacks.moeva2.objective_calculator import ObjectiveCalculator

        classifier = Classifier(
            load_model(self.estimator._parameters["paths"]["model"])
        )
        scaler = self.estimator._scaler
        objective_calc = ObjectiveCalculator(
            classifier,
            self.estimator._constraints,
            minimize_class=1,
            thresholds=self.estimator._parameters["thresholds"],
            min_max_scaler=scaler,
            ml_scaler=scaler,
        )

        x_np = scaler.inverse_transform(x.numpy())
        x_init_np = scaler.inverse_transform(x_init.numpy())
        if len(x.shape) == 2:
            x_np = x_np[:, np.newaxis, :]
            x_init_np = x_init_np[:, np.newaxis, :]

        success_rates = objective_calc.success_rate_3d(x_init_np, x_np)

        print(success_rates)

    def _compute_tf(
        self,
        x: "tf.Tensor",
        x_init: "tf.Tensor",
        y: "tf.Tensor",
        mask: "tf.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        random_init: bool,
    ) -> "tf.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation = (
                random_sphere(n, m, eps, self.norm)
                .reshape(x.shape)
                .astype(ART_NUMPY_DTYPE)
            )
            random_perturbation = tf.convert_to_tensor(random_perturbation)
            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation(x_adv, y, mask)

        if "adaptive_eps_step" in self.loss_evaluation:
            iteration_per_step = self.max_iter // 7
            current_power = np.float64(self.iter_counter // iteration_per_step + 1)
            eps_step_dynamic = eps * (1 / np.float_power(10.0, current_power))
            print(f"{current_power}: {eps_step_dynamic}")
        else:
            eps_step_dynamic = eps_step

        if self.estimator.experiment is not None:
            self.estimator.experiment.log_metric(
                "eta",
                eps_step_dynamic,
                step=self.iter_counter,
                epoch=0,
            )
        self.iter_counter = self.iter_counter + 1

        x_adv = self._apply_perturbation(x_adv, perturbation, eps_step_dynamic)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = tf.add(perturbation, x_init)

        # Update the features of x according to the constraints
        if "repair" in self.loss_evaluation:
            x_adv = self.fix_features_type(x_adv)

        # evaluate success rate with constraints

        # if self._i_max_iter%10==0 and self._batch_id==0:
        #    self._compute_multi_objectives(x_adv, x_init)

        return x_adv

    def fix_features_type(self, x):

        constraints = self.estimator._constraints
        unscaled = self.estimator.unscale_features(x)

        unscaled = constraints.fix_features_types(unscaled)

        return self.estimator.scale_features(unscaled)

    def _compute_perturbation(  # pylint: disable=W0221
        self, x: "tf.Tensor", y: "tf.Tensor", mask: Optional["tf.Tensor"]
    ) -> "tf.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :return: Perturbations.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad: tf.Tensor = self.estimator.loss_gradient(
            x, y, iter_i=self._i_max_iter, batch_id=self._batch_id, targeted=self.targeted
        )

        # Write summary
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "gradients/norm-L1/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=1),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-L2/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=2),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-Linf/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=np.inf),
                global_step=self._i_max_iter,
            )

            if hasattr(self.estimator, "compute_losses"):
                losses = self.estimator.compute_losses(x=x, y=y)

                for key, value in losses.items():
                    self.summary_writer.add_scalar(
                        "loss/{}/batch-{}".format(key, self._batch_id),
                        np.mean(value),
                        global_step=self._i_max_iter,
                    )

        # Check for NaN before normalisation an replace with 0
        if tf.reduce_any(tf.math.is_nan(grad)):
            logger.warning(
                "Elements of the loss gradient are NaN and have been replaced with 0.0."
            )
            grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)

        # Apply mask
        if mask is not None:
            grad = tf.where(mask == 0.0, 0.0, grad)

        # Apply norm bound
        if self.norm == np.inf:
            grad = tf.sign(grad)

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = tf.divide(
                grad, (tf.math.reduce_sum(tf.abs(grad), axis=ind, keepdims=True) + tol)
            )

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = tf.divide(
                grad,
                (
                    tf.math.sqrt(
                        tf.math.reduce_sum(
                            tf.math.square(grad), axis=ind, keepdims=True
                        )
                    )
                    + tol
                ),
            )

        assert x.shape == grad.shape

        return grad
