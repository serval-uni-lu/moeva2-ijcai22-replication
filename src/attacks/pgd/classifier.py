from art.classifiers import TensorFlowV2Classifier
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from src.attacks.moeva2.constraints import Constraints
from sklearn.preprocessing import MinMaxScaler
from comet_ml import Experiment
from art.config import ART_NUMPY_DTYPE


class TF2Classifier(TensorFlowV2Classifier):
    def __init__(
        self,
        model: Callable,
        nb_classes: int,
        input_shape: Tuple[int, ...],
        loss_object: Optional["tf.keras.losses.Loss"] = None,
        constraints: Constraints = None,
        scaler: MinMaxScaler = None,
        experiment: Experiment = None,
        experiment_batch_skip: int = 8,
        parameters: dict = None,
        train_step: Optional[Callable] = None,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union[
            "Preprocessor", List["Preprocessor"], None
        ] = None,
        postprocessing_defences: Union[
            "Postprocessor", List["Postprocessor"], None
        ] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Initialization specific to TensorFlow v2 models.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :param nb_classes: the number of classes in the classification task.
        :param input_shape: shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
            the model and computing gradients of the loss w.r.t. the input.
        :type loss_object: `tf.keras.losses`
        :param constraints: The use case specific constraint object
        :param scaler the Scaler used for the Model features
        :param train_step: A function that applies a gradient update to the trainable variables with signature
                           train_step(model, images, labels).
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """

        super().__init__(
            model,
            nb_classes,
            input_shape,
            loss_object,
            train_step,
            channels_first,
            clip_values,
            preprocessing_defences,
            postprocessing_defences,
            preprocessing,
        )

        self._constraints = constraints
        self._scaler = scaler
        self._parameters = parameters
        self._experiment = experiment
        self.experiment = experiment
        self._randomindex = np.random.randint(constraints.get_nb_constraints())
        self._experiment_batch_skip = experiment_batch_skip
        self.history = []
        self.last_iter = 0

    def unscale_features(self, inputs):
        inputs -= self._scaler.min_
        inputs /= self._scaler.scale_

        return inputs

    def scale_features(self, inputs):
        inputs *= self._scaler.scale_
        inputs += self._scaler.min_

        return inputs

    def constraint_loss(self, inputs):
        violations = []

        # scaled_inputs = self._scaler.inverse_transform(inputs).astype('float32')

        inputs = self.unscale_features(inputs)

        violations = self._constraints.evaluate(inputs, use_tensors=True)

        return violations

    def loss_gradient(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "tf.Tensor"],
        y: Union[np.ndarray, "tf.Tensor"],
        training_mode: bool = False,
        targeted: bool = False,
        batch_id: int = 0,
        iter_i: int = 0,
        wc: float = 0.1,
        **kwargs
    ) -> Union[np.ndarray, "tf.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :param batch_id: Identifier of the current batch.
        :param iter_i: Identifier of the current multistep attack iteration.
        :param wc: weight of the flip loss.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]
        self.last_iter = iter_i
        if self._loss_object is None:
            raise TypeError(
                "The loss function `loss_object` is required for computing loss gradients, but it has not been "
                "defined."
            )

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                # Apply preprocessing
                if self.all_framework_preprocessing:
                    x_grad = tf.convert_to_tensor(x)
                    tape.watch(x_grad)
                    x_input, y_input = self._apply_preprocessing(x_grad, y=y, fit=False)
                else:
                    x_preprocessed, y_preprocessed = self._apply_preprocessing(
                        x, y=y, fit=False
                    )
                    x_grad = tf.convert_to_tensor(x_preprocessed)
                    tape.watch(x_grad)
                    x_input = x_grad
                    y_input = y_preprocessed

                predictions = self.model(x_input, training=training_mode)

                if self._reduce_labels:
                    loss_class = self._loss_object(
                        np.argmax(y_input, axis=1), predictions
                    )
                else:
                    loss_class = self._loss_object(y_input, predictions)

                loss_constraints = self.constraint_loss(x_input)

                # Different options for the value of the constraints loss
                # - alternate
                # - single (with ID)
                # - Else sum
                constraints_optim = self._parameters.get("constraints_optim")
                if "alt_constraints" in constraints_optim:
                    nb_constraints = loss_constraints.shape[1]
                    loss_constraints_reduced = loss_constraints[
                        :, iter_i % nb_constraints
                    ]

                elif "single_constraints" in constraints_optim:
                    ctr_id = self._parameters.get("ctr_id")
                    loss_constraints_reduced = loss_constraints[:, ctr_id]

                else:
                    loss_constraints_reduced = tf.reduce_sum(loss_constraints, 1)

                # Logging experiments metrics
                if self._experiment and batch_id % self._experiment_batch_skip == 0:
                    self._experiment.log_metric(
                        "loss_constraints_max",
                        loss_constraints_reduced.numpy().max(),
                        step=iter_i,
                        epoch=batch_id,
                    )
                    self._experiment.log_metric(
                        "loss_flip_max",
                        loss_class.numpy().max(),
                        step=iter_i,
                        epoch=batch_id,
                    )

                    self._experiment.log_metric(
                        "loss_constraints_mean",
                        loss_constraints_reduced.numpy().mean(),
                        step=iter_i,
                        epoch=batch_id,
                    )
                    self._experiment.log_metric(
                        "loss_flip_mean",
                        loss_class.numpy().mean(),
                        step=iter_i,
                        epoch=batch_id,
                    )
                    if "full" in self._parameters.get("save_history"):
                        for i in range(loss_constraints.shape[1]):
                            constraint_loss = loss_constraints[:, i].numpy().mean()
                            self._experiment.log_metric(
                                "ctr_{}".format(i),
                                constraint_loss,
                                step=iter_i,
                                epoch=batch_id,
                            )

                loss_class = loss_class * tf.constant(
                    1 - 2 * int(targeted), dtype=ART_NUMPY_DTYPE
                )
                # we want to minimize the loss not increased it
                loss_constraints_reduced_history = loss_constraints_reduced
                loss_constraints_reduced = loss_constraints_reduced * tf.constant(
                    -1, dtype=ART_NUMPY_DTYPE
                )

                # Strategy to combine class loss and constraints loss
                # Flip half the time then constraints only
                # Alternate n times constraints n time flip, n = alternate_frequency
                # Constraints + flip all the time weighted
                # Only constraints
                # Only flip
                loss_evaluation = self._parameters.get("loss_evaluation")
                if "constraints+flip+manual" in loss_evaluation:
                    # total_iterations = self._parameters.get("budget", 100)/2
                    if iter_i < 100:
                        loss = loss_class
                    else:
                        loss = loss_constraints_reduced
                elif "constraints+flip+constraints" in loss_evaluation:
                    total_iterations = self._parameters.get("nb_iter", 100)
                    if iter_i < total_iterations / 2:
                        loss = loss_class
                    else:
                        loss = loss_constraints_reduced
                elif "constraints+flip+alternate" in loss_evaluation:
                    alternate_frequency = self._parameters.get("alternate_frequency", 5)
                    loss = (
                        loss_class
                        if (iter_i // alternate_frequency) % 2
                        else loss_constraints_reduced
                    )
                elif "constraints+flip" in loss_evaluation:
                    loss = loss_class + loss_constraints_reduced
                elif "constraints" in loss_evaluation:
                    loss = loss_constraints_reduced
                else:
                    loss = loss_class

            if self._experiment and batch_id % self._experiment_batch_skip == 0:
                self._experiment.log_metric(
                    "loss_max",
                    loss.numpy().max(),
                    step=iter_i,
                    epoch=batch_id,
                )

                self._experiment.log_metric(
                    "loss_mean",
                    loss.numpy().mean(),
                    step=iter_i,
                    epoch=batch_id,
                )

            if "full" in self._parameters.get("save_history"):
                self.history.append(
                    np.column_stack(
                        [
                            loss,
                            loss_class,
                            loss_constraints_reduced_history,
                            loss_constraints,
                        ]
                    )
                )
            if "reduced" in self._parameters.get("save_history"):
                self.history.append(
                    np.column_stack(
                        [
                            loss,
                            loss_class,
                            loss_constraints_reduced_history,
                        ]
                    )
                )

            gradients = tape.gradient(loss, x_grad)

            if isinstance(x, np.ndarray):
                gradients = gradients.numpy()

        else:
            raise NotImplementedError("Expecting eager execution.")

        # Apply preprocessing gradients
        if not self.all_framework_preprocessing:
            gradients = self._apply_preprocessing_gradient(x, gradients)

        if isinstance(gradients, np.ndarray):
            if self._experiment and batch_id % self._experiment_batch_skip == 0:
                self._experiment.log_metric(
                    "grad_max", gradients.max(), step=iter_i, epoch=batch_id
                )
                self._experiment.log_metric(
                    "grad_mean", gradients.mean(), step=iter_i, epoch=batch_id
                )
                self._experiment.log_metric(
                    "grad_min", gradients.min(), step=iter_i, epoch=batch_id
                )
        else:
            if self._experiment and batch_id % self._experiment_batch_skip == 0:
                self._experiment.log_metric(
                    "grad_max", gradients.numpy().max(), step=iter_i, epoch=batch_id
                )
                self._experiment.log_metric(
                    "grad_mean", gradients.numpy().mean(), step=iter_i, epoch=batch_id
                )
                self._experiment.log_metric(
                    "grad_min", gradients.numpy().min(), step=iter_i, epoch=batch_id
                )
        return gradients

    def compute_loss(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "tf.Tensor"],
        y: Union[np.ndarray, "tf.Tensor"],
        reduction: str = "none",
        training_mode: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Compute the loss.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of losses of the same shape as `x`.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if self._loss_object is None:
            raise TypeError("The loss function `loss_object` is required for computing losses, but it is not defined.")
        prev_reduction = self._loss_object.reduction
        if reduction == "none":
            self._loss_object.reduction = tf.keras.losses.Reduction.NONE
        elif reduction == "mean":
            self._loss_object.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        elif reduction == "sum":
            self._loss_object.reduction = tf.keras.losses.Reduction.SUM

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y, fit=False)

        if tf.executing_eagerly():
            x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)
            predictions = self.model(x_preprocessed_tf, training=training_mode)
            if self._reduce_labels:
                loss_class = self._loss_object(np.argmax(y, axis=1), predictions)
            else:
                loss_class = self._loss_object(y, predictions)

            wc = 0.1
            loss_constraints = self.constraint_loss(x_preprocessed)
            constraints_optim = self._parameters.get("constraints_optim")
            if "single_constraints" in constraints_optim:
                ctr_id = self._parameters.get("ctr_id")
                loss_constraints_reduced = loss_constraints[:, ctr_id]
            else:
                loss_constraints_reduced = tf.reduce_sum(loss_constraints, 1)

            loss_constraints_reduced = loss_constraints_reduced * tf.constant(
                -1, dtype=ART_NUMPY_DTYPE
            )

            if reduction == "mean":
                loss_constraints_reduced = tf.reduce_mean(loss_constraints_reduced)

            loss_evaluation = self._parameters.get("loss_evaluation")
            if "constraints+flip+manual" in loss_evaluation:
                # total_iterations = self._parameters.get("budget", 100)/2
                if self.last_iter < 100:
                    loss = loss_class
                else:
                    loss = loss_constraints_reduced
            elif "constraints+flip" in loss_evaluation:
                loss = wc * loss_class + loss_constraints_reduced
            elif "constraints" in loss_evaluation:
                loss = loss_constraints_reduced
            else:
                loss = loss_class

        else:
            raise NotImplementedError("Expecting eager execution.")

        self._loss_object.reduction = prev_reduction
        return loss.numpy()
