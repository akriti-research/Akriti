"""One-step finite-difference gradient search for three loss weights.

Each objective evaluation retrains the U-Net from scratch with the same seeds,
then scores mean validation Dice. The search is deliberately bounded to one
central-difference gradient step because each evaluation is a full experiment.
It returns the best weight vector tested; it does not guarantee a global optimum.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


RUNNER = Path(__file__).with_name("run_isic.py")
COMPONENTS = ("lovasz", "boundary_weighted_bce", "focal")


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def softmax_weights(logits: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray([logits[0], logits[1], 0.0], dtype=np.float64)
    values -= np.max(values)
    exponentials = np.exp(values)
    weights = exponentials / exponentials.sum()
    return tuple(float(value) for value in weights)


def weight_key(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(value, 8) for value in weights)


def read_metrics(path: Path, seeds: tuple[int, ...], epochs: int) -> tuple[float, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training did not produce its metrics file: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_seed = {int(row["seed"]): row for row in rows}
    if set(by_seed) != set(seeds):
        raise RuntimeError(
            f"Expected metrics for seeds {seeds}, found {tuple(sorted(by_seed))}."
        )
    scores: dict[str, float] = {}
    for seed in seeds:
        row = by_seed[seed]
        if int(row["epochs_completed"]) != epochs:
            raise RuntimeError(
                f"Seed {seed} completed {row['epochs_completed']} of {epochs} requested epochs."
            )
        if row["diverged"].strip().lower() != "false":
            raise RuntimeError(f"Seed {seed} diverged; rejecting this objective evaluation.")
        scores[str(seed)] = float(row["Dice"])
    return 1.0 - float(np.mean(list(scores.values()))), scores


def append_history(path: Path, row: dict[str, Any]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize Lovasz/BW-BCE/Focal mixture weights by one finite-difference "
            "gradient step in softmax-logit coordinates through full model retraining."
        )
    )
    parser.add_argument("--images-dir", type=Path, default=Path("./input/images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("./input/masks"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=parse_seeds, default=(42, 123, 2024))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--boundary-tolerance", type=float, default=2.0)
    parser.add_argument("--finite-difference-step", type=float, default=0.5)
    parser.add_argument("--gradient-step-size", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/three_loss_search"))
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if args.finite_difference_step <= 0.0 or args.gradient_step_size < 0.0:
        raise ValueError("Finite-difference step must be positive; gradient step cannot be negative.")
    if args.boundary_tolerance < 0.0:
        raise ValueError("Boundary tolerance cannot be negative.")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"Expected the training runner beside this file: {RUNNER}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "optimizer_history.csv"
    if history_path.exists():
        raise FileExistsError(
            f"Search history already exists at {history_path}; choose a fresh --output-dir."
        )
    cache: dict[tuple[float, float, float], dict[str, Any]] = {}
    trial_index = 0

    def evaluate(logits: np.ndarray, label: str) -> dict[str, Any]:
        nonlocal trial_index
        weights = softmax_weights(logits)
        key = weight_key(weights)
        if key in cache:
            return cache[key]

        trial_root = args.output_dir / f"trial_{trial_index:02d}_{label}"
        trial_index += 1
        weight_argument = ",".join(f"{value:.10f}" for value in weights)
        command = [
            sys.executable,
            str(RUNNER),
            "--images-dir",
            str(args.images_dir),
            "--masks-dir",
            str(args.masks_dir),
            "--mode",
            "full",
            "--loss",
            "combined_three",
            "--loss-weights",
            weight_argument,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--seeds",
            ",".join(str(seed) for seed in args.seeds),
            "--split-seed",
            str(args.split_seed),
            "--boundary-tolerance",
            str(args.boundary_tolerance),
            "--output-dir",
            str(trial_root),
        ]
        if args.deterministic:
            command.append("--deterministic")
        print(
            f"\nSearch evaluation {trial_index}/6 ({label}): "
            f"Lovasz={weights[0]:.3f}, BW-BCE={weights[1]:.3f}, Focal={weights[2]:.3f}"
        )
        subprocess.run(command, check=True)

        metrics_path = trial_root / "full" / "combined_three" / "metrics_per_seed.csv"
        objective, dice_by_seed = read_metrics(metrics_path, args.seeds, args.epochs)
        row = {
            "evaluation": trial_index - 1,
            "label": label,
            "weight_lovasz": weights[0],
            "weight_boundary_weighted_bce": weights[1],
            "weight_focal": weights[2],
            "mean_validation_dice": 1.0 - objective,
            "objective_1_minus_mean_validation_dice": objective,
            "dice_by_seed_json": json.dumps(dice_by_seed, sort_keys=True),
            "trial_output": str(trial_root),
        }
        append_history(history_path, row)
        result = {
            "weights": weights,
            "objective": objective,
            "dice_by_seed": dice_by_seed,
            "trial_output": str(trial_root),
            "label": label,
        }
        cache[key] = result
        return result

    center = np.array([0.0, 0.0], dtype=np.float64)
    step = args.finite_difference_step
    probes = [
        evaluate(center, "center"),
        evaluate(center + np.array([step, 0.0]), "plus_lovasz_direction"),
        evaluate(center - np.array([step, 0.0]), "minus_lovasz_direction"),
        evaluate(center + np.array([0.0, step]), "plus_bce_direction"),
        evaluate(center - np.array([0.0, step]), "minus_bce_direction"),
    ]

    gradient = np.array(
        [
            (probes[1]["objective"] - probes[2]["objective"]) / (2.0 * step),
            (probes[3]["objective"] - probes[4]["objective"]) / (2.0 * step),
        ],
        dtype=np.float64,
    )
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > 1e-12 and args.gradient_step_size > 0.0:
        candidate_logits = center - args.gradient_step_size * gradient / gradient_norm
    else:
        candidate_logits = center.copy()
    candidate = evaluate(candidate_logits, "gradient_candidate")
    evaluated = [*probes, candidate]
    best = min(evaluated, key=lambda item: item["objective"])

    summary = {
        "method": "one_step_central_finite_difference_gradient_descent_softmax_logits",
        "primary_objective": "1 - mean validation Dice across the fixed seeds",
        "seeds": list(args.seeds),
        "epochs_per_evaluation": args.epochs,
        "finite_difference_step_in_logit_space": step,
        "normalized_gradient_step_size": args.gradient_step_size,
        "gradient_in_logit_space": gradient.tolist(),
        "gradient_norm": gradient_norm,
        "candidate_weights": dict(zip(COMPONENTS, candidate["weights"])),
        "best_tested_weights": dict(zip(COMPONENTS, best["weights"])),
        "best_tested_mean_validation_dice": 1.0 - best["objective"],
        "best_trial_output": best["trial_output"],
        "evaluations": [
            {
                "label": item["label"],
                "weights": dict(zip(COMPONENTS, item["weights"])),
                "mean_validation_dice": 1.0 - item["objective"],
                "trial_output": item["trial_output"],
            }
            for item in evaluated
        ],
        "interpretation": (
            "Best tested mixture in this bounded, one-step search. It is not a global "
            "optimum; validation Dice is used for both checkpoint selection and weight search."
        ),
        "official_test_data_used": False,
    }
    summary_path = args.output_dir / "optimizer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nBest tested weights:")
    for component, weight in zip(COMPONENTS, best["weights"]):
        print(f"  {component}: {weight:.4f}")
    print(f"Mean validation Dice: {1.0 - best['objective']:.4f}")
    print(f"Search summary: {summary_path}")
    print("This is the best tested candidate, not a guaranteed global optimum.")


if __name__ == "__main__":
    main()
