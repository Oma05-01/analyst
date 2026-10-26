"""Combine Football Analyst Agent backtest CSVs and analyse paired performance.

Run from the repository root:
    python -m src.analyze_backtests

The script reads only the per-match CSVs written by src.evaluate_models. It does
not connect to or modify PostgreSQL. It reports per-season and pooled metrics,
cluster-bootstrap confidence intervals for paired score differences, and fixed-
width reliability bins for 1X2 probabilities.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DEFAULT_INPUTS = [
    "reports/backtest_2021-2022.csv",
    "reports/backtest_2022-2023.csv",
    "reports/backtest_2023-2024.csv",
    "reports/backtest_2024-2025.csv",
]
METRICS = ("brier", "rps", "log_loss")
PROBABILITY_COLUMNS = {
    "dixon_coles_v2": {
        "HOME": "dc_p_home",
        "DRAW": "dc_p_draw",
        "AWAY": "dc_p_away",
    },
    "baseline_poisson_v1": {
        "HOME": "baseline_p_home",
        "DRAW": "baseline_p_draw",
        "AWAY": "baseline_p_away",
    },
}


def load_reports(paths: list[str]) -> pd.DataFrame:
    required = {
        "match_id", "season", "match_date_utc", "home_goals", "away_goals",
        "dc_p_home", "dc_p_draw", "dc_p_away",
        "baseline_p_home", "baseline_p_draw", "baseline_p_away",
        "dc_brier", "dc_rps", "dc_log_loss",
        "baseline_brier", "baseline_rps", "baseline_log_loss",
    }
    frames = []
    for path_str in paths:
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Backtest report not found: {path}")
        frame = pd.read_csv(path, dtype={"season": str, "match_date_utc": str})
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"Backtest report is empty: {path}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["season"] = combined["season"].astype(str)
    combined["match_date_utc"] = combined["match_date_utc"].astype(str)
    if combined.duplicated(["season", "match_id"]).any():
        duplicates = combined.loc[
            combined.duplicated(["season", "match_id"], keep=False),
            ["season", "match_id"],
        ]
        raise ValueError(
            "Duplicate fixture rows across the supplied reports:\n"
            + duplicates.to_string(index=False)
        )

    numeric_columns = [
        "home_goals", "away_goals",
        *[f"{prefix}_p_{side}" for prefix in ("dc", "baseline")
          for side in ("home", "draw", "away")],
        *[f"{prefix}_{metric}" for prefix in ("dc", "baseline") for metric in METRICS],
    ]
    for column in numeric_columns:
        combined[column] = pd.to_numeric(combined[column], errors="raise")
    if not np.isfinite(combined[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("A report contains a NaN or infinite numeric value")

    for prefix in ("dc", "baseline"):
        p_cols = [f"{prefix}_p_{s}" for s in ("home", "draw", "away")]
        p = combined[p_cols].to_numpy(dtype=float)
        if (p < 0).any() or (p > 1).any():
            raise ValueError(f"Invalid probabilities in columns {p_cols}")
        sums = p.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6, rtol=0):
            bad = np.flatnonzero(np.abs(sums - 1.0) > 1e-6)[:5]
            raise ValueError(
                f"{prefix} probabilities do not sum to one; example row indexes: {bad.tolist()}"
            )

    # Reconstruct outcomes from the actual goals rather than relying on text labels.
    combined["actual_outcome"] = np.select(
        [combined["home_goals"] > combined["away_goals"],
         combined["home_goals"] == combined["away_goals"]],
        ["HOME", "DRAW"], default="AWAY",
    )
    return combined


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, part in [(s, g) for s, g in frame.groupby("season", sort=True)]:
        for model_name, prefix in (("dixon_coles_v2", "dc"),
                                   ("baseline_poisson_v1", "baseline")):
            rows.append({
                "scope": scope,
                "model": model_name,
                "n": len(part),
                **{metric: float(part[f"{prefix}_{metric}"].mean()) for metric in METRICS},
            })
    for model_name, prefix in (("dixon_coles_v2", "dc"),
                               ("baseline_poisson_v1", "baseline")):
        rows.append({
            "scope": "ALL_SEASONS",
            "model": model_name,
            "n": len(frame),
            **{metric: float(frame[f"{prefix}_{metric}"].mean()) for metric in METRICS},
        })
    return pd.DataFrame(rows)


def _difference_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        metric: (frame[f"dc_{metric}"].to_numpy(dtype=float)
                 - frame[f"baseline_{metric}"].to_numpy(dtype=float))
        for metric in METRICS
    }


def bootstrap_scope(
    frame: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
    scope_name: str,
) -> list[dict]:
    """Cluster bootstrap dates, preserving all fixtures on a sampled date.

    For ALL_SEASONS, dates are resampled separately within each season so every
    season remains represented. The bootstrap resamples date clusters, not
    individual matches, because fixtures evaluated on one date share a fitted model.
    """
    if scope_name == "ALL_SEASONS":
        season_frames = [g for _, g in frame.groupby("season", sort=True)]
    else:
        season_frames = [frame[frame["season"] == scope_name]]

    clusters_by_season: list[list[pd.DataFrame]] = []
    point_parts = []
    for season_frame in season_frames:
        clusters = [g for _, g in season_frame.groupby("match_date_utc", sort=True)]
        if not clusters:
            continue
        clusters_by_season.append(clusters)
        point_parts.append(season_frame)
    if not point_parts:
        return []

    point = pd.concat(point_parts, ignore_index=True)
    point_diff = _difference_arrays(point)
    point_estimates = {metric: float(values.mean()) for metric, values in point_diff.items()}
    samples = {metric: np.empty(reps, dtype=float) for metric in METRICS}

    cluster_payloads = []
    for clusters in clusters_by_season:
        payload = []
        for cluster in clusters:
            diffs = _difference_arrays(cluster)
            sums = np.array([diffs[m].sum() for m in METRICS], dtype=float)
            payload.append((sums, len(cluster)))
        cluster_sums = np.vstack([x[0] for x in payload])
        cluster_counts = np.array([x[1] for x in payload], dtype=float)
        cluster_payloads.append((cluster_sums, cluster_counts))

    for rep in range(reps):
        total_sums = np.zeros(len(METRICS), dtype=float)
        total_count = 0.0
        for cluster_sums, cluster_counts in cluster_payloads:
            chosen = rng.integers(0, len(cluster_counts), size=len(cluster_counts))
            total_sums += cluster_sums[chosen].sum(axis=0)
            total_count += cluster_counts[chosen].sum()
        for i, metric in enumerate(METRICS):
            samples[metric][rep] = total_sums[i] / total_count

    rows = []
    baseline_means = {
        metric: float(point[f"baseline_{metric}"].mean()) for metric in METRICS
    }
    for metric in METRICS:
        lo, hi = np.quantile(samples[metric], [0.025, 0.975])
        delta = point_estimates[metric]
        base = baseline_means[metric]
        rows.append({
            "scope": scope_name,
            "metric": metric,
            "n": len(point),
            "mean_difference_dc_minus_baseline": delta,
            "ci95_lower": float(lo),
            "ci95_upper": float(hi),
            "baseline_mean": base,
            "relative_score_reduction_pct": (-delta / base * 100.0) if base else np.nan,
            "bootstrap_reps": reps,
            "cluster_unit": "match_date_utc",
        })
    return rows


def calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, outcome_columns in PROBABILITY_COLUMNS.items():
        for outcome, column in outcome_columns.items():
            probabilities = frame[column].to_numpy(dtype=float)
            observed = (frame["actual_outcome"].to_numpy() == outcome).astype(float)
            for bin_idx in range(10):
                low, high = bin_idx / 10.0, (bin_idx + 1) / 10.0
                mask = (probabilities >= low) & (
                    (probabilities < high) if bin_idx < 9 else (probabilities <= high)
                )
                n = int(mask.sum())
                if n == 0:
                    continue
                mean_predicted = float(probabilities[mask].mean())
                observed_rate = float(observed[mask].mean())
                rows.append({
                    "model": model_name,
                    "outcome": outcome,
                    "probability_bin": f"{low:.1f}-{high:.1f}",
                    "n": n,
                    "mean_predicted_probability": mean_predicted,
                    "observed_frequency": observed_rate,
                    "absolute_calibration_gap": abs(mean_predicted - observed_rate),
                })
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", default=DEFAULT_INPUTS,
        help="per-match CSV reports (default: four reports from 2021-22 through 2024-25)",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--seed", type=int, default=20261023)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.bootstrap_reps < 100:
        print("--bootstrap-reps must be at least 100", file=sys.stderr)
        return 2
    try:
        frame = load_reports(args.inputs)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)

        summary = metric_summary(frame)
        bootstrap_rows = []
        for season in sorted(frame["season"].unique()):
            bootstrap_rows.extend(bootstrap_scope(frame, args.bootstrap_reps, rng, season))
        bootstrap_rows.extend(bootstrap_scope(frame, args.bootstrap_reps, rng, "ALL_SEASONS"))
        bootstrap = pd.DataFrame(bootstrap_rows)
        calibration = calibration_table(frame)

        summary_path = output_dir / "backtest_combined_summary.csv"
        bootstrap_path = output_dir / "backtest_paired_uncertainty.csv"
        calibration_path = output_dir / "backtest_calibration.csv"
        summary.to_csv(summary_path, index=False, float_format="%.10g")
        bootstrap.to_csv(bootstrap_path, index=False, float_format="%.10g")
        calibration.to_csv(calibration_path, index=False, float_format="%.10g")

        print("FOUR-SEASON BACKTEST SUMMARY")
        print(f"Reports: {len(args.inputs)} | paired fixtures: {len(frame)} | seasons: {', '.join(sorted(frame['season'].unique()))}")
        print("\nMean scores by season and across all seasons (lower is better):")
        print(summary.round(5).to_string(index=False))
        print("\nPaired score differences with 95% date-cluster bootstrap intervals")
        print("Negative differences favour Dixon-Coles. Intervals describe resampling uncertainty within these seasons; they do not guarantee future-season performance.")
        display = bootstrap[[
            "scope", "metric", "n", "mean_difference_dc_minus_baseline",
            "ci95_lower", "ci95_upper", "relative_score_reduction_pct",
        ]].copy()
        print(display.round(5).to_string(index=False))
        print("\nCalibration bins saved. Compare mean_predicted_probability with observed_frequency; closer values indicate better calibration in that bin.")
        print(f"Summary:     {summary_path}")
        print(f"Uncertainty: {bootstrap_path}")
        print(f"Calibration: {calibration_path}")
        return 0
    except Exception as exc:
        print(f"Backtest analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
