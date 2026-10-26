#!/usr/bin/env python3
"""Build leakage-aware historical starting-XI strength features from FPL history.

This is a FEATURE-BUILDING step only. It does not fit a match-outcome model and
it does not modify PostgreSQL or locked production predictions.

For each fixture, starting-player xG+xA per 90 estimates are calculated from
strictly earlier UTC calendar dates. Estimates are shrunk toward a position
rate learned from earlier appearances. A second feature measures the current
XI's estimated attacking contribution relative to that club's recent XIs.

Defaults:
    input:  fpl_complete_fixture_history.csv
    output: data/processed/player_lineup_features.csv

Example:
    python -m src.build_player_lineup_features
    python -m src.build_player_lineup_features --input path/to/fpl_complete_fixture_history.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("build_player_lineup_features")
REQUIRED_COLUMNS = {
    "season", "player_code", "element_id", "player_name", "position",
    "player_team", "fixture_id", "kickoff_utc", "was_home", "minutes",
    "started", "xG", "xA",
}
DEFAULT_OUTPUT_SEASONS = ("2022-23", "2023-24", "2024-25")
VALID_POSITIONS = {"GKP", "DEF", "MID", "FWD"}


def normalise_season(value) -> str | None:
    """Normalise '2022-23', '2022- 23', or '2022-2023' to 'YYYY-YY'."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{2}|\d{4})", s)
    if not m:
        return None
    start = int(m.group(1))
    tail = m.group(2)
    end = int(tail) if len(tail) == 4 else (start // 100) * 100 + int(tail)
    if end == start + 1:
        return f"{start}-{end % 100:02d}"
    return None


def season_long(short_season: str) -> str:
    start, short_end = short_season.split("-")
    return f"{start}-{int(start[:2] + short_end)}"


def parse_bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "1.0", "yes", "y"}:
        return True
    if s in {"false", "0", "0.0", "no", "n"}:
        return False
    return None


def clean_string(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def player_key(row) -> str:
    code = row.player_code
    if pd.notna(code):
        try:
            return f"code:{int(float(code))}"
        except (ValueError, TypeError, OverflowError):
            pass
    name = clean_string(row.player_name).casefold()
    if name:
        return f"name:{name}"
    season = clean_string(row.season)
    element = row.element_id
    if pd.notna(element):
        return f"element:{season}:{int(float(element))}"
    return f"unknown:{season}:{row.Index}"


def team_key(name) -> str:
    return clean_string(name).casefold()


def estimate_player_rate(row, player_state, position_state, shrinkage_minutes, min_position_minutes):
    """Return (smoothed xG+xA per 90, has_player_history, estimated_prior_minutes)."""
    pos = clean_string(row.position).upper()
    p_state = player_state.get(player_key(row), {"xgi": 0.0, "minutes": 0.0})
    player_minutes = float(p_state["minutes"])
    has_player_history = player_minutes > 0.0
    if pos not in VALID_POSITIONS:
        return np.nan, has_player_history, player_minutes

    pos_state = position_state.get(pos, {"xgi": 0.0, "minutes": 0.0})
    pos_minutes = float(pos_state["minutes"])
    if pos_minutes < min_position_minutes or pos_minutes <= 0:
        return np.nan, has_player_history, player_minutes

    position_rate = 90.0 * float(pos_state["xgi"]) / pos_minutes
    denominator = player_minutes + shrinkage_minutes
    if denominator <= 0:
        return position_rate, has_player_history, player_minutes
    smoothed_rate = 90.0 * (
        float(p_state["xgi"]) + (position_rate * shrinkage_minutes / 90.0)
    ) / denominator
    return max(0.0, float(smoothed_rate)), has_player_history, player_minutes


def unique_team_name(rows) -> str | None:
    names = [clean_string(v) for v in rows["player_team"].dropna().tolist()]
    names = [n for n in names if n]
    if not names:
        return None
    counts = pd.Series(names).value_counts()
    return str(counts.index[0])


def load_history(path: Path, output_seasons: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Player history CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Player history is missing required columns: " + ", ".join(missing))

    df["source_season"] = df["season"].astype(str)
    df["season_short"] = df["season"].map(normalise_season)
    if df["season_short"].isna().any():
        bad = sorted(df.loc[df["season_short"].isna(), "source_season"].unique().tolist())
        LOG.warning("Ignoring rows with unrecognised season labels: %s", bad[:10])
    df["season_long"] = df["season_short"].map(lambda s: season_long(s) if s else None)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df["match_date_utc"] = df["kickoff_utc"].dt.floor("D")
    df["was_home_bool"] = df["was_home"].map(parse_bool)
    df["started_bool"] = df["started"].map(parse_bool)
    for col in ("minutes", "xG", "xA", "fixture_id", "player_code", "element_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["position"] = df["position"].map(lambda x: clean_string(x).upper())
    df["player_team"] = df["player_team"].map(clean_string)
    df["player_name"] = df["player_name"].map(clean_string)
    df["xgi"] = df["xG"] + df["xA"]

    before = len(df)
    df = df[df["season_short"].notna() & df["kickoff_utc"].notna() & df["fixture_id"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        LOG.warning("Ignored %d rows with invalid season, kickoff, or fixture id.", dropped)

    # The entire dataset is processed chronologically to build history. Only
    # output_seasons are emitted; older seasons can contribute when xG/xA exist.
    df = df.sort_values(["match_date_utc", "season_short", "fixture_id"], kind="mergesort")
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame, output_seasons: set[str], half_life_days: float,
                   shrinkage_minutes: float, min_position_minutes: float,
                   team_baseline_matches: int, team_history_window: int) -> pd.DataFrame:
    player_state: dict[str, dict[str, float]] = {}
    position_state: dict[str, dict[str, float]] = {
        p: {"xgi": 0.0, "minutes": 0.0} for p in VALID_POSITIONS
    }
    team_recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=team_history_window))
    last_date = None
    output_rows = []
    decay_lambda = math.log(2.0) / half_life_days

    for match_date, day_rows in df.groupby("match_date_utc", sort=True):
        if pd.isna(match_date):
            continue
        if last_date is not None:
            delta_days = max(0.0, (match_date - last_date).total_seconds() / 86400.0)
            factor = math.exp(-decay_lambda * delta_days)
            for state in player_state.values():
                state["xgi"] *= factor
                state["minutes"] *= factor
            for state in position_state.values():
                state["xgi"] *= factor
                state["minutes"] *= factor
        last_date = match_date

        day_team_updates = []
        day_rows_to_emit = []
        for (src_season, fixture_id), fixture_rows in day_rows.groupby(["season_short", "fixture_id"], sort=False):
            if src_season not in output_seasons:
                # Still derive historical team lineup-strength summaries for
                # later seasons; no output is needed for this fixture.
                emit = False
            else:
                emit = True

            home_rows = fixture_rows[fixture_rows["was_home_bool"].eq(True)]
            away_rows = fixture_rows[fixture_rows["was_home_bool"].eq(False)]
            home_name = unique_team_name(home_rows)
            away_name = unique_team_name(away_rows)
            kickoff_values = fixture_rows["kickoff_utc"].dropna()
            kickoff = kickoff_values.min() if not kickoff_values.empty else pd.NaT

            home_starters = home_rows[home_rows["started_bool"].eq(True)]
            away_starters = away_rows[away_rows["started_bool"].eq(True)]
            home_xi_valid = len(home_starters) == 11 and home_name is not None
            away_xi_valid = len(away_starters) == 11 and away_name is not None

            def lineup_metrics(starters, valid):
                if not valid:
                    return {
                        "total": np.nan, "prior_count": 0, "prior_share": 0.0,
                        "imputed_count": 0, "names_json": json.dumps(
                            starters["player_name"].tolist(), ensure_ascii=False
                        ), "player_rates": [],
                    }
                rates = []
                prior_count = 0
                for row in starters.itertuples():
                    rate, had_history, _prior_mins = estimate_player_rate(
                        row, player_state, position_state, shrinkage_minutes, min_position_minutes
                    )
                    rates.append(rate)
                    prior_count += int(had_history)
                total = float(np.sum(rates)) if len(rates) == 11 and all(np.isfinite(r) for r in rates) else np.nan
                return {
                    "total": total,
                    "prior_count": prior_count,
                    "prior_share": prior_count / 11.0,
                    "imputed_count": 11 - prior_count,
                    "names_json": json.dumps(starters["player_name"].tolist(), ensure_ascii=False),
                    "player_rates": rates,
                }

            home_metrics = lineup_metrics(home_starters, home_xi_valid)
            away_metrics = lineup_metrics(away_starters, away_xi_valid)
            h_team_key = team_key(home_name) if home_name else ""
            a_team_key = team_key(away_name) if away_name else ""

            def team_delta(name_key_value, current_total):
                hist = list(team_recent.get(name_key_value, [])) if name_key_value else []
                if not np.isfinite(current_total) or len(hist) < team_baseline_matches:
                    return np.nan, len(hist), (float(np.mean(hist)) if hist else np.nan)
                recent_mean = float(np.mean(hist[-team_history_window:]))
                return float(current_total - recent_mean), len(hist), recent_mean

            h_delta, h_hist_n, h_recent_mean = team_delta(h_team_key, home_metrics["total"])
            a_delta, a_hist_n, a_recent_mean = team_delta(a_team_key, away_metrics["total"])
            features_ready = bool(np.isfinite(home_metrics["total"]) and np.isfinite(away_metrics["total"]))
            delta_ready = bool(np.isfinite(h_delta) and np.isfinite(a_delta))

            if emit:
                day_rows_to_emit.append({
                    "season": season_long(src_season),
                    "source_season": src_season,
                    "fixture_id": int(fixture_id),
                    "kickoff_utc": kickoff.isoformat() if pd.notna(kickoff) else None,
                    "match_date_utc": match_date.date().isoformat(),
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_starters_json": home_metrics["names_json"],
                    "away_starters_json": away_metrics["names_json"],
                    "home_xi_starters_count": len(home_starters),
                    "away_xi_starters_count": len(away_starters),
                    "home_xi_player_history_count": home_metrics["prior_count"],
                    "away_xi_player_history_count": away_metrics["prior_count"],
                    "home_xi_player_history_share": home_metrics["prior_share"],
                    "away_xi_player_history_share": away_metrics["prior_share"],
                    "home_xi_imputed_player_count": home_metrics["imputed_count"],
                    "away_xi_imputed_player_count": away_metrics["imputed_count"],
                    "home_xi_attack_xgi_per90_sum": home_metrics["total"],
                    "away_xi_attack_xgi_per90_sum": away_metrics["total"],
                    "home_xi_attack_delta_vs_recent_team_mean": h_delta,
                    "away_xi_attack_delta_vs_recent_team_mean": a_delta,
                    "home_team_prior_lineups_used": h_hist_n,
                    "away_team_prior_lineups_used": a_hist_n,
                    "home_recent_xi_attack_mean": h_recent_mean,
                    "away_recent_xi_attack_mean": a_recent_mean,
                    "lineup_strength_features_ready": features_ready,
                    "lineup_delta_features_ready": delta_ready,
                    "feature_note": (
                        "xG+xA per 90; all player and position statistics use only earlier UTC calendar dates"
                    ),
                })

            # Update club lineup history only after all current-date features have
            # been computed, preventing same-day fixtures leaking into each other.
            if home_name and np.isfinite(home_metrics["total"]):
                day_team_updates.append((h_team_key, float(home_metrics["total"])))
            if away_name and np.isfinite(away_metrics["total"]):
                day_team_updates.append((a_team_key, float(away_metrics["total"])))

        # Apply team updates after all fixtures for this UTC date were evaluated.
        for key, total in day_team_updates:
            team_recent[key].append(total)

        # Update player and positional history after the full date has been
        # scored, so no appearance from the current date can affect its features.
        for row in day_rows.itertuples():
            mins = float(row.minutes) if pd.notna(row.minutes) else 0.0
            xgi = float(row.xgi) if pd.notna(row.xgi) else np.nan
            pos = clean_string(row.position).upper()
            if mins <= 0.0 or not np.isfinite(xgi):
                continue
            key = player_key(row)
            state = player_state.setdefault(key, {"xgi": 0.0, "minutes": 0.0})
            state["xgi"] += max(0.0, xgi)
            state["minutes"] += mins
            if pos in VALID_POSITIONS:
                position_state[pos]["xgi"] += max(0.0, xgi)
                position_state[pos]["minutes"] += mins

        output_rows.extend(day_rows_to_emit)

    result = pd.DataFrame(output_rows)
    if result.empty:
        raise ValueError("No feature rows were produced. Check season labels and input data.")
    result = result.sort_values(["season", "kickoff_utc", "fixture_id"], kind="mergesort").reset_index(drop=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="fpl_complete_fixture_history.csv", help="input player-fixture history CSV")
    parser.add_argument("--output", default="data/processed/player_lineup_features.csv", help="output feature CSV")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_OUTPUT_SEASONS),
                        help="seasons to emit, e.g. 2022-23 2023-24 2024-25; all input seasons still warm history")
    parser.add_argument("--half-life-days", type=float, default=180.0,
                        help="half-life for decaying player/position performance history (default 180)")
    parser.add_argument("--shrinkage-minutes", type=float, default=450.0,
                        help="minutes of positional prior used to shrink player rates (default 450)")
    parser.add_argument("--min-position-prior-minutes", type=float, default=450.0,
                        help="minimum decayed minutes before a position prior can be used (default 450)")
    parser.add_argument("--team-baseline-matches", type=int, default=3,
                        help="minimum previous team lineups needed for lineup-delta feature (default 3)")
    parser.add_argument("--team-history-window", type=int, default=5,
                        help="number of previous team lineups for baseline (default 5)")
    args = parser.parse_args()

    if args.half_life_days <= 0 or args.shrinkage_minutes < 0 or args.min_position_prior_minutes < 0:
        parser.error("half-life must be positive and shrinkage/prior minutes must be non-negative")
    if args.team_baseline_matches < 1 or args.team_history_window < args.team_baseline_matches:
        parser.error("team history window must be >= team baseline matches >= 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_seasons = {normalise_season(s) for s in args.seasons}
    if None in output_seasons:
        parser.error("Unrecognised season label in --seasons. Use labels like 2023-24.")

    try:
        df = load_history(Path(args.input), output_seasons)
        target_count = df["season_short"].isin(output_seasons).sum()
        print("PLAYER LINEUP FEATURE BUILD (offline; no database writes)")
        print(f"Input rows available for chronological history: {len(df):,}")
        print(f"Target seasons: {', '.join(sorted(output_seasons))}")
        print(f"Target-season player-fixture rows: {target_count:,}")
        print(f"Player/position history half-life: {args.half_life_days:g} days")
        print(f"Player-rate shrinkage: {args.shrinkage_minutes:g} prior minutes")

        features = build_features(
            df, output_seasons, args.half_life_days, args.shrinkage_minutes,
            args.min_position_prior_minutes, args.team_baseline_matches, args.team_history_window,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output, index=False)

        print("\nFEATURE SUMMARY")
        features["both_teams_exactly_11"] = (
            features["home_xi_starters_count"].eq(11)
            & features["away_xi_starters_count"].eq(11)
        )
        summary = features.groupby("season", dropna=False).agg(
            fixtures=("fixture_id", "nunique"),
            lineup_strength_ready=("lineup_strength_features_ready", "sum"),
            lineup_delta_ready=("lineup_delta_features_ready", "sum"),
            both_teams_exactly_11=("both_teams_exactly_11", "sum"),
            home_mean_player_history_share=("home_xi_player_history_share", "mean"),
            away_mean_player_history_share=("away_xi_player_history_share", "mean"),
        )
        print(summary.round(3).to_string())
        print(f"\nOutput saved: {output}")
        print("Leakage guard: all player and position performance from a UTC calendar date is added only after that date's features are calculated.")
        print("Interpretation: xG+xA is a contribution proxy, not proof of injury or unavailability. This file does not fit or apply a prediction model.")
        return 0
    except Exception as exc:
        LOG.exception("Feature build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
