"""Plot reliability diagrams from src.analyze_backtests output.

Run from the repository root:
    python -m src.plot_calibration

Input is the aggregated calibration table produced by:
    python -m src.analyze_backtests --bootstrap-reps 5000

Creates one image per outcome (HOME, DRAW, AWAY) comparing Dixon-Coles and
baseline predictions, plus a small CSV summary of weighted absolute calibration
gaps. This script reads local report files only; it does not access PostgreSQL.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = {
    "model",
    "outcome",
    "probability_bin",
    "n",
    "mean_predicted_probability",
    "observed_frequency",
}
MODEL_LABELS = {
    "dixon_coles_v2": "Dixon-Coles v2",
    "baseline_poisson_v1": "League-average baseline",
}
OUTCOMES = ("HOME", "DRAW", "AWAY")


def load_calibration(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration CSV not found: {path}\n"
            "Run `python -m src.analyze_backtests --bootstrap-reps 5000` first."
        )
    df = pd.read_csv(path)
    missing = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Calibration CSV is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"Calibration CSV is empty: {path}")

    for column in (
        "n", "mean_predicted_probability", "observed_frequency"
    ):
        df[column] = pd.to_numeric(df[column], errors="raise")
    if (df["n"] <= 0).any():
        raise ValueError("Every calibration bin must have n > 0")
    probability_values = df[["mean_predicted_probability", "observed_frequency"]].to_numpy(dtype=float)
    if not np.isfinite(probability_values).all():
        raise ValueError("Calibration CSV contains non-finite probabilities")
    if ((probability_values < 0) | (probability_values > 1)).any():
        raise ValueError("Calibration values must lie in [0, 1]")
    if not set(df["outcome"].unique()).issubset(set(OUTCOMES)):
        raise ValueError("Unexpected outcome labels in calibration CSV")
    return df


def weighted_calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted absolute calibration gap (ECE-style descriptive metric)."""
    rows = []
    for (model, outcome), group in df.groupby(["model", "outcome"], sort=True):
        total_n = int(group["n"].sum())
        gaps = (
            group["n"].to_numpy(dtype=float)
            * np.abs(
                group["mean_predicted_probability"].to_numpy(dtype=float)
                - group["observed_frequency"].to_numpy(dtype=float)
            )
        )
        rows.append({
            "model": model,
            "outcome": outcome,
            "n": total_n,
            "weighted_absolute_calibration_gap": float(gaps.sum() / total_n),
        })
    return pd.DataFrame(rows)


def plot_outcome(df: pd.DataFrame, outcome: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for reliability plots. Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    plotted = 0
    for model, group in df[df["outcome"] == outcome].groupby("model", sort=True):
        group = group.sort_values("mean_predicted_probability")
        if group.empty:
            continue
        label = MODEL_LABELS.get(model, model)
        # Marker size grows slowly with bin count to make sparse bins apparent.
        sizes = 24 + 70 * np.sqrt(group["n"].to_numpy(dtype=float) / group["n"].max())
        ax.plot(
            group["mean_predicted_probability"],
            group["observed_frequency"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=label,
        )
        ax.scatter(
            group["mean_predicted_probability"],
            group["observed_frequency"],
            s=sizes,
            alpha=0.75,
        )
        plotted += len(group)

    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"No calibration bins available for outcome {outcome}")

    ax.set_title(f"Reliability diagram — {outcome.title()} outcome")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed outcome frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/backtest_calibration.csv")
    parser.add_argument("--output-dir", default="reports/calibration_plots")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        calibration = load_calibration(Path(args.input))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = weighted_calibration_summary(calibration)
        summary_path = output_dir / "calibration_gap_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.8g")

        for outcome in OUTCOMES:
            plot_path = output_dir / f"reliability_{outcome.lower()}.png"
            plot_outcome(calibration, outcome, plot_path)
            print(f"Saved: {plot_path}")

        print("\nWeighted absolute calibration gaps (lower is closer on average; descriptive only):")
        print(summary.round(4).to_string(index=False))
        print(f"\nSummary saved: {summary_path}")
        print("Note: bins can hide within-bin variation, and sparse bins are noisy. Do not fit a calibration correction from this evaluation set and score that correction on the same data.")
        return 0
    except Exception as exc:
        print(f"Calibration plotting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
