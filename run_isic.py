"""ISIC 2016 segmentation loss optimization runner.

The data loader and U-Net preserve the supplied experiment notebooks. This
runner intentionally uses only the supplied 900-image training partition for
the 720/180 train/validation split; it has no official-test data argument.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
# Python 3.12 removed stdlib distutils. Setuptools supplies the compatibility
# module TensorFlow 2.18 imports, so load its shim before importing TensorFlow.
import setuptools
import tensorflow as tf
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


IMAGE_SIZE = 128
EXPECTED_PAIRS = 900
TRAIN_COUNT = 720
VAL_COUNT = 180
DEFAULT_SEEDS = (42, 123, 2024)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".bmp")


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            print("Deterministic TensorFlow ops could not be enabled in this runtime.")


def image_id(path: str | Path) -> str:
    stem = Path(path).stem
    return re.sub(r"_(segmentation|mask)$", "", stem, flags=re.IGNORECASE).lower()


def files_in(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {folder}")
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: p.name.lower(),
    )


def unique_id_lookup(paths: list[Path], label: str) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        key = image_id(path)
        if key in lookup:
            duplicates.append(key)
        lookup[key] = path
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs: {sorted(set(duplicates))[:10]}")
    return lookup


def load_and_split(
    images_dir: Path, masks_dir: Path, split_seed: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    image_files = files_in(images_dir, IMAGE_EXTENSIONS)
    mask_files = files_in(masks_dir, MASK_EXTENSIONS)
    image_lookup = unique_id_lookup(image_files, "image")
    mask_lookup = unique_id_lookup(mask_files, "mask")
    image_ids, mask_ids = set(image_lookup), set(mask_lookup)
    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask IDs do not match. "
            f"Images without masks: {missing_masks[:10]}; "
            f"masks without images: {missing_images[:10]}"
        )
    if len(image_files) != EXPECTED_PAIRS or len(mask_files) != EXPECTED_PAIRS:
        raise ValueError(
            f"Expected exactly {EXPECTED_PAIRS} matched images and masks from the "
            f"official training partition; found {len(image_files)} images and "
            f"{len(mask_files)} masks. Do not point these paths at a train+test mix."
        )

    pairs = [(image_lookup[key], mask_lookup[key], key) for key in sorted(image_ids)]
    np.random.default_rng(split_seed).shuffle(pairs)

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ordered_ids: list[str] = []
    for image_path, mask_path, key in pairs:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
            )
            masks.append((np.asarray(mask) > 127).astype(np.float32)[..., None])
        ordered_ids.append(key)

    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(masks, dtype=np.float32)
    if x.shape != (EXPECTED_PAIRS, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected image tensor shape: {x.shape}")
    if y.shape != (EXPECTED_PAIRS, IMAGE_SIZE, IMAGE_SIZE, 1):
        raise ValueError(f"Unexpected mask tensor shape: {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values were found in the loaded data.")
    if float(y.mean()) <= 0.0 or float(y.mean()) > 0.95:
        raise ValueError(
            "Mask foreground fraction looks invalid (all black or over 95% white). "
            "Check that lesion pixels are white and background is black."
        )

    split_at = int(0.8 * len(x))
    if split_at != TRAIN_COUNT or len(x) - split_at != VAL_COUNT:
        raise AssertionError("The deterministic split did not produce 720/180 images.")
    return x, y, ordered_ids[:split_at], ordered_ids[split_at:]


def write_split_manifest(path: Path, train_ids: list[str], val_ids: list[str]) -> None:
    rows = [(key, "train") for key in train_ids] + [(key, "validation") for key in val_ids]
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))
        expected = [{"image_id": key, "split": split} for key, split in rows]
        if existing != expected:
            raise ValueError(
                f"The split manifest at {path} differs from the current seed-42 split. "
                "Use a new output directory rather than silently changing the split."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("image_id", "split"))
        writer.writeheader()
        writer.writerows({"image_id": key, "split": split} for key, split in rows)


def _flatten_batch(tensor: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tensor, [tf.shape(tensor)[0], -1])


def lovasz_hinge(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    """Per-image Lovasz hinge, preserving the supplied probability-to-logit path."""
    clip = tf.constant(1e-6, dtype=tf.float32)
    p = tf.clip_by_value(_flatten_batch(probabilities), clip, 1.0 - clip)
    g = _flatten_batch(tf.cast(y_true, tf.float32))
    logits = tf.math.log(p / (1.0 - p))
    errors = 1.0 - logits * (2.0 * g - 1.0)
    errors_sorted, permutation = tf.math.top_k(errors, k=tf.shape(errors)[1])
    g_sorted = tf.gather(g, permutation, batch_dims=1)
    positives = tf.reduce_sum(g, axis=1, keepdims=True)
    intersection = positives - tf.cumsum(g_sorted, axis=1)
    union = positives + tf.cumsum(1.0 - g_sorted, axis=1)
    jaccard = 1.0 - tf.math.divide_no_nan(intersection, union)
    gradient = tf.concat(
        [jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], axis=1
    )
    per_image = tf.reduce_sum(
        tf.nn.relu(errors_sorted) * tf.stop_gradient(gradient), axis=1
    )
    return tf.reduce_mean(per_image)


def _signed_distance_numpy(masks: np.ndarray) -> np.ndarray:
    result = np.zeros(masks.shape, dtype=np.float32)
    for index, mask in enumerate(masks[..., 0] > 0.5):
        if mask.any() and not mask.all():
            result[index, ..., 0] = distance_transform_edt(~mask) - distance_transform_edt(mask)
    return result


def signed_distance(masks: tf.Tensor) -> tf.Tensor:
    distance = tf.numpy_function(_signed_distance_numpy, [masks], tf.float32)
    distance.set_shape(masks.shape)
    return distance


def boundary_weighted_bce(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    clip = tf.constant(1e-7, dtype=tf.float32)
    p = tf.clip_by_value(probabilities, clip, 1.0 - clip)
    pixel_bce = -(
        y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p)
    )
    distance = tf.abs(signed_distance(y_true))
    weight = 1.0 + 5.0 * tf.exp(-tf.square(distance) / (2.0 * 3.0**2))
    return tf.reduce_sum(weight * pixel_bce) / tf.reduce_sum(weight)


def dice_loss(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    smooth = tf.constant(1e-6, dtype=tf.float32)
    axes = [1, 2, 3]
    intersection = tf.reduce_sum(y_true * probabilities, axis=axes)
    total = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(probabilities, axis=axes)
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (total + smooth))


def tversky_loss(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    alpha, beta, smooth = 0.3, 0.7, 1e-6
    axes = [1, 2, 3]
    true_positive = tf.reduce_sum(y_true * probabilities, axis=axes)
    false_positive = tf.reduce_sum((1.0 - y_true) * probabilities, axis=axes)
    false_negative = tf.reduce_sum(y_true * (1.0 - probabilities), axis=axes)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - tf.reduce_mean(score)


def binary_cross_entropy(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    p = tf.clip_by_value(probabilities, 1e-7, 1.0 - 1e-7)
    return -tf.reduce_mean(
        y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p)
    )


def focal_loss(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    p = tf.clip_by_value(probabilities, 1e-7, 1.0 - 1e-7)
    p_t = y_true * p + (1.0 - y_true) * (1.0 - p)
    return -tf.reduce_mean(tf.pow(1.0 - p_t, 2.0) * tf.math.log(p_t))


def boundary_dice_loss(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    def inner_band(values: tf.Tensor) -> tf.Tensor:
        eroded = 1.0 - tf.nn.max_pool2d(
            1.0 - values, ksize=5, strides=1, padding="SAME"
        )
        return values - eroded

    predicted_band = inner_band(probabilities)
    true_band = inner_band(y_true)
    axes = [1, 2, 3]
    intersection = tf.reduce_sum(predicted_band * true_band, axis=axes)
    total = tf.reduce_sum(predicted_band, axis=axes) + tf.reduce_sum(true_band, axis=axes)
    return 1.0 - tf.reduce_mean((2.0 * intersection + 1e-6) / (total + 1e-6))


def signed_distance_map_loss(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    target = tf.clip_by_value(signed_distance(y_true) / 10.0, -1.0, 1.0)
    return tf.reduce_mean(tf.square((1.0 - 2.0 * probabilities) - target))


BASELINE_LOSSES = {
    "lovasz_hinge": lovasz_hinge,
    "boundary_weighted_bce": boundary_weighted_bce,
    "dice": dice_loss,
    "tversky": tversky_loss,
    "bce": binary_cross_entropy,
    "focal": focal_loss,
    "boundary_dice": boundary_dice_loss,
    "sdm": signed_distance_map_loss,
}
LOSS_CHOICES = ("combined", "combined_three", *BASELINE_LOSSES.keys())
THREE_LOSS_COMPONENTS = ("lovasz", "boundary_weighted_bce", "focal")


def parse_loss_weights(value: str) -> tuple[float, float, float]:
    try:
        weights = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Loss weights must be three comma-separated numbers."
        ) from error
    if len(weights) != 3 or any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise argparse.ArgumentTypeError(
            "Loss weights must be three finite, nonnegative numbers."
        )
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise argparse.ArgumentTypeError("Three loss weights must sum to 1.")
    return weights


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)


def build_unet() -> tf.keras.Model:
    """The supplied small U-Net: 16/32/64 encoder, 128 bottleneck, sigmoid head."""
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPooling2D()(c3)
    bottleneck = conv_block(p3, 128)
    c4 = conv_block(tf.keras.layers.Concatenate()([tf.keras.layers.UpSampling2D()(bottleneck), c3]), 64)
    c5 = conv_block(tf.keras.layers.Concatenate()([tf.keras.layers.UpSampling2D()(c4), c2]), 32)
    c6 = conv_block(tf.keras.layers.Concatenate()([tf.keras.layers.UpSampling2D()(c5), c1]), 16)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c6)
    return tf.keras.Model(inputs, outputs, name="Small_UNet")


def per_image_dice(y_true: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    truth = tf.cast(y_true >= 0.5, tf.float32)
    prediction = tf.cast(probabilities >= 0.5, tf.float32)
    axes = [1, 2, 3]
    intersection = tf.reduce_sum(truth * prediction, axis=axes)
    total = tf.reduce_sum(truth, axis=axes) + tf.reduce_sum(prediction, axis=axes)
    score = tf.math.divide_no_nan(2.0 * intersection, total)
    return tf.where(total > 0.0, score, tf.ones_like(score))


class ExperimentModel(tf.keras.Model):
    def __init__(
        self,
        network: tf.keras.Model,
        objective: str,
        alpha: float,
        loss_weights: tuple[float, float, float],
    ):
        super().__init__(name="ISIC_loss_experiment")
        self.network = network
        self.objective = objective
        self.alpha = None
        self.objective_weights: dict[str, float]
        if objective == "combined":
            self.alpha = self.add_weight(
                name="alpha",
                shape=(),
                initializer=tf.keras.initializers.Constant(alpha),
                trainable=False,
            )
            self.component_functions = {
                "lovasz": lovasz_hinge,
                "boundary_weighted_bce": boundary_weighted_bce,
            }
            self.objective_weights = {
                "lovasz": float(alpha),
                "boundary_weighted_bce": float(1.0 - alpha),
            }
        elif objective == "combined_three":
            self.component_functions = {
                "lovasz": lovasz_hinge,
                "boundary_weighted_bce": boundary_weighted_bce,
                "focal": focal_loss,
            }
            self.objective_weights = dict(zip(THREE_LOSS_COMPONENTS, loss_weights))
        else:
            self.component_functions = {objective: BASELINE_LOSSES[objective]}
            self.objective_weights = {objective: 1.0}
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dice_tracker = tf.keras.metrics.Mean(name="dice")
        self.component_trackers = {
            key: tf.keras.metrics.Mean(name=f"{key}_loss")
            for key in self.component_functions
        }
        self.weighted_component_trackers = {
            key: tf.keras.metrics.Mean(name=f"weighted_{key}_loss")
            for key in self.component_functions
        }
        self.gradient_trackers = {
            "grad_norm_total": tf.keras.metrics.Mean(name="grad_norm_total"),
            **{
                f"grad_norm_{key}": tf.keras.metrics.Mean(name=f"grad_norm_{key}")
                for key in self.component_functions
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.network(inputs, training=training)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.loss_tracker,
            self.dice_tracker,
            *self.component_trackers.values(),
            *self.weighted_component_trackers.values(),
        ]

    def component_losses(
        self, y_true: tf.Tensor, probabilities: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        return {
            key: function(y_true, probabilities)
            for key, function in self.component_functions.items()
        }

    def combine(self, components: dict[str, tf.Tensor]) -> tf.Tensor:
        if self.objective == "combined":
            return self.alpha * components["lovasz"] + (1.0 - self.alpha) * components[
                "boundary_weighted_bce"
            ]
        if self.objective == "combined_three":
            return tf.add_n(
                [
                    tf.cast(self.objective_weights[key], value.dtype) * value
                    for key, value in components.items()
                ]
            )
        return components[self.objective]

    def component_weight(self, key: str) -> float | tf.Tensor:
        if self.objective == "combined":
            if key == "lovasz":
                return self.alpha
            return 1.0 - self.alpha
        return self.objective_weights[key]

    def current_weights(self) -> dict[str, float]:
        if self.objective == "combined":
            alpha = float(self.alpha.numpy())
            return {"lovasz": alpha, "boundary_weighted_bce": 1.0 - alpha}
        return dict(self.objective_weights)

    def _metric_results(self, include_gradients: bool = False) -> dict[str, tf.Tensor]:
        result = {metric.name: metric.result() for metric in self.metrics}
        if include_gradients:
            result.update(
                {metric.name: metric.result() for metric in self.gradient_trackers.values()}
            )
        return result

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        images, labels = data
        batch_size = tf.cast(tf.shape(images)[0], tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            probabilities = self.network(images, training=True)
            components = self.component_losses(labels, probabilities)
            total_loss = self.combine(components)
        variables = self.network.trainable_variables
        total_gradients = tape.gradient(total_loss, variables)
        total_norm = tf.linalg.global_norm(
            [gradient for gradient in total_gradients if gradient is not None]
        )
        self.gradient_trackers["grad_norm_total"].update_state(
            total_norm, sample_weight=batch_size
        )
        for key, value in components.items():
            self.component_trackers[key].update_state(value, sample_weight=batch_size)
            self.weighted_component_trackers[key].update_state(
                value * self.component_weight(key), sample_weight=batch_size
            )
            component_gradients = tape.gradient(value, variables)
            component_norm = tf.linalg.global_norm(
                [gradient for gradient in component_gradients if gradient is not None]
            )
            self.gradient_trackers[f"grad_norm_{key}"].update_state(
                component_norm, sample_weight=batch_size
            )
        del tape
        self.optimizer.apply_gradients(
            [
                (gradient, variable)
                for gradient, variable in zip(total_gradients, variables)
                if gradient is not None
            ]
        )
        self.loss_tracker.update_state(total_loss, sample_weight=batch_size)
        self.dice_tracker.update_state(
            per_image_dice(labels, probabilities), sample_weight=batch_size
        )
        return self._metric_results(include_gradients=True)

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        images, labels = data
        batch_size = tf.cast(tf.shape(images)[0], tf.float32)
        probabilities = self.network(images, training=False)
        components = self.component_losses(labels, probabilities)
        total_loss = self.combine(components)
        self.loss_tracker.update_state(total_loss, sample_weight=batch_size)
        self.dice_tracker.update_state(
            per_image_dice(labels, probabilities), sample_weight=batch_size
        )
        for key, value in components.items():
            self.component_trackers[key].update_state(value, sample_weight=batch_size)
            self.weighted_component_trackers[key].update_state(
                value * self.component_weight(key), sample_weight=batch_size
            )
        return self._metric_results()


def projected_alpha_update(alpha: float, gradient: float, learning_rate: float) -> float:
    """Project one supplied scalar gradient step onto the interval [0, 1]."""
    return float(np.clip(alpha - learning_rate * gradient, 0.0, 1.0))


class GradientNormReset(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        for metric in self.model.gradient_trackers.values():
            metric.reset_state()


class ProjectedAlphaCallback(tf.keras.callbacks.Callback):
    """Optional direct validation-loss update with model weights held fixed."""

    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self.learning_rate <= 0.0 or logs is None:
            return
        lovasz = logs.get("val_lovasz_loss")
        boundary_bce = logs.get("val_boundary_weighted_bce_loss")
        if lovasz is None or boundary_bce is None:
            return
        gradient = float(lovasz) - float(boundary_bce)
        previous = float(self.model.alpha.numpy())
        updated = projected_alpha_update(previous, gradient, self.learning_rate)
        self.model.alpha.assign(updated)
        logs["alpha_used_for_epoch"] = previous
        logs["alpha_direct_val_gradient"] = gradient
        logs["alpha_next_epoch"] = updated


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, border_value=1)


def boundary_scores(
    truth: np.ndarray, prediction: np.ndarray, tolerance: float
) -> tuple[float, float]:
    truth_empty = not truth.any()
    prediction_empty = not prediction.any()
    if truth_empty and prediction_empty:
        return 1.0, 0.0
    if truth_empty or prediction_empty:
        return 0.0, float("nan")
    true_boundary = binary_boundary(truth)
    predicted_boundary = binary_boundary(prediction)
    if not true_boundary.any() and not predicted_boundary.any():
        return 1.0, 0.0
    if not true_boundary.any() or not predicted_boundary.any():
        return 0.0, float("nan")
    predicted_to_true = distance_transform_edt(~true_boundary)[predicted_boundary]
    true_to_predicted = distance_transform_edt(~predicted_boundary)[true_boundary]
    precision = float(np.mean(predicted_to_true <= tolerance))
    recall = float(np.mean(true_to_predicted <= tolerance))
    boundary_f1 = (
        0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    )
    hd95 = max(
        float(np.percentile(predicted_to_true, 95)),
        float(np.percentile(true_to_predicted, 95)),
    )
    return boundary_f1, hd95


def evaluate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, tolerance: float
) -> dict[str, float | int]:
    true_masks = y_true[..., 0] >= 0.5
    predicted_masks = probabilities[..., 0] >= 0.5
    dice_values: list[float] = []
    iou_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    boundary_f1_values: list[float] = []
    hd95_values: list[float] = []

    for truth, prediction in zip(true_masks, predicted_masks):
        tp = int(np.logical_and(truth, prediction).sum())
        fp = int(np.logical_and(~truth, prediction).sum())
        fn = int(np.logical_and(truth, ~prediction).sum())
        truth_count = tp + fn
        prediction_count = tp + fp
        union = tp + fp + fn
        if truth_count == 0 and prediction_count == 0:
            dice_values.append(1.0)
            iou_values.append(1.0)
            precision_values.append(1.0)
            recall_values.append(1.0)
        else:
            dice_values.append(2.0 * tp / (2.0 * tp + fp + fn))
            iou_values.append(tp / union if union else 1.0)
            precision_values.append(tp / prediction_count if prediction_count else 0.0)
            recall_values.append(tp / truth_count if truth_count else 0.0)
        bf1, hd95 = boundary_scores(truth, prediction, tolerance)
        boundary_f1_values.append(bf1)
        if np.isfinite(hd95):
            hd95_values.append(hd95)

    return {
        "Dice": float(np.mean(dice_values)),
        "IoU": float(np.mean(iou_values)),
        "Precision": float(np.mean(precision_values)),
        "Recall": float(np.mean(recall_values)),
        "BoundaryF1": float(np.mean(boundary_f1_values)),
        "HD95_px": float(np.mean(hd95_values)) if hd95_values else float("nan"),
        "HD95_undefined_images": len(y_true) - len(hd95_values),
    }


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("./input/images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("./input/masks"))
    parser.add_argument("--mode", choices=("sanity", "full"), default="sanity")
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="combined")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--loss-weights",
        type=parse_loss_weights,
        default="0.333333,0.333333,0.333334",
        help=(
            "For combined_three only: Lovasz,BW-BCE,Focal weights, nonnegative "
            "and summing to 1."
        ),
    )
    parser.add_argument("--alpha-update-lr", type=float, default=0.0)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--boundary-tolerance", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--sanity-train-samples", type=int, default=8)
    parser.add_argument("--sanity-val-samples", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if args.alpha_update_lr < 0.0:
        raise ValueError("alpha-update-lr cannot be negative.")
    if args.alpha_update_lr > 0.0 and args.loss != "combined":
        raise ValueError("Projected alpha updates apply only to --loss combined.")
    if args.loss == "combined_three" and not np.isclose(
        sum(args.loss_weights), 1.0, atol=1e-6
    ):
        raise ValueError("The three combined loss weights must sum to 1.")
    if args.boundary_tolerance < 0.0:
        raise ValueError("Boundary tolerance cannot be negative.")

    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    print("Loading the 900-image training partition and reproducing the seed-42 split...")
    x, y, train_ids, val_ids = load_and_split(
        args.images_dir, args.masks_dir, args.split_seed
    )
    output_root = args.output_dir / args.mode
    output_root.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output_root / "split_manifest.csv", train_ids, val_ids)
    print(f"Loaded {len(x)} pairs; train/validation = {len(train_ids)}/{len(val_ids)}")
    print(f"Input shape: {x.shape}; mask shape: {y.shape}; lesion fraction: {float(y.mean()):.4f}")

    train_x, train_y = x[:TRAIN_COUNT], y[:TRAIN_COUNT]
    val_x, val_y = x[TRAIN_COUNT:], y[TRAIN_COUNT:]
    if args.mode == "sanity":
        train_n = min(args.sanity_train_samples, len(train_x))
        val_n = min(args.sanity_val_samples, len(val_x))
        train_x, train_y = train_x[:train_n], train_y[:train_n]
        val_x, val_y = val_x[:val_n], val_y[:val_n]
        epochs = 1
        seeds = args.seeds[:1]
        print(
            f"SANITY MODE: {len(train_x)} train + {len(val_x)} validation images, "
            f"1 epoch, seed {seeds[0]}. No full experiment will run."
        )
    else:
        epochs = args.epochs
        seeds = args.seeds

    history_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n=== {args.loss} | seed {seed} | {epochs} epoch(s) ===")
        set_seed(seed, deterministic=args.deterministic)
        tf.keras.backend.clear_session()
        model = ExperimentModel(
            build_unet(), args.loss, args.alpha, args.loss_weights
        )
        model(tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32), training=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate, clipnorm=1.0
            ),
            run_eagerly=False,
            # Boundary-weighted BCE uses tf.numpy_function for SciPy distance maps.
            # That operation is not XLA-compatible; the source notebooks also set this False.
            jit_compile=False,
        )
        seed_dir = output_root / args.loss / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_dir / "best.weights.h5"
        callbacks: list[tf.keras.callbacks.Callback] = [
            GradientNormReset(),
            ProjectedAlphaCallback(args.alpha_update_lr),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_dice",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        start_time = time.perf_counter()
        history = model.fit(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            batch_size=args.batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )
        elapsed = time.perf_counter() - start_time
        if not checkpoint_path.exists():
            raise RuntimeError(f"No best-validation checkpoint was written: {checkpoint_path}")
        model.load_weights(checkpoint_path)
        probabilities = model.predict(val_x, batch_size=args.batch_size, verbose=0)
        metrics = evaluate_metrics(val_y, probabilities, args.boundary_tolerance)
        metric_rows.append(
            {
                "loss": args.loss,
                "seed": seed,
                **metrics,
                "epochs_completed": len(history.epoch),
                "best_checkpoint": str(checkpoint_path),
                "training_seconds": elapsed,
                "alpha_final": (
                    float(model.alpha.numpy()) if model.alpha is not None else ""
                ),
                "loss_weights_json": json.dumps(model.current_weights(), sort_keys=True),
                "diverged": not all(
                    np.isfinite(value)
                    for values in history.history.values()
                    for value in values
                ),
            }
        )
        for epoch_index in range(len(history.epoch)):
            row: dict[str, Any] = {"loss": args.loss, "seed": seed, "epoch": epoch_index + 1}
            for name, values in history.history.items():
                row[name] = values[epoch_index]
            history_rows.append(row)
        print(
            f"Seed {seed}: val Dice={metrics['Dice']:.4f}, "
            f"IoU={metrics['IoU']:.4f}, Boundary F1={metrics['BoundaryF1']:.4f}, "
            f"HD95={metrics['HD95_px']:.3f}px"
        )

    append_csv(output_root / args.loss / "history.csv", history_rows)
    append_csv(output_root / args.loss / "metrics_per_seed.csv", metric_rows)
    config = {
        "mode": args.mode,
        "loss": args.loss,
        "epochs_requested": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "alpha_initial": args.alpha,
        "alpha_update_lr": args.alpha_update_lr,
        "loss_weights": model.current_weights() if metric_rows else {},
        "seeds": list(seeds),
        "split_seed": args.split_seed,
        "boundary_tolerance_pixels_at_128": args.boundary_tolerance,
        "train_count_used": len(train_x),
        "validation_count_used": len(val_x),
        "official_test_data_used": False,
    }
    config_path = output_root / args.loss / "run_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nSaved run files under: {output_root / args.loss}")
    if args.mode == "sanity":
        print("Review the checkpoint, CSV files, and printed metrics before requesting a full run.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
