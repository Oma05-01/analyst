#!/usr/bin/env python3
"""Generate an experimental Dixon-Coles forecast using a saved lineup snapshot.

This script is deliberately separate from the production prediction pipeline.
It reads PostgreSQL, the player history/features, and the append-only lineup
snapshots. Forecasts are printed by default and appended to a separate JSONL
file only when --save is supplied. It never updates predictions, scores, or
match_previews in PostgreSQL.

Before first use, rebuild historical lineup features to include recent seasons:
  python -m src.build_player_lineup_features --seasons 2022-23 2023-24 2024-25 2025-26 2026-27

Commands:
  python -m src.generate_lineup_forecast --list-upcoming
  python -m src.generate_lineup_forecast --match-id 1234 --lineup-type projected
  python -m src.generate_lineup_forecast --match-id 1234 --lineup-type official_lineup --save
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from src.db_engine import get_sqlalchemy_engine
from src.models.dixon_coles import DixonColesModel, UnknownTeamError, outcome_probs, score_matrix
from src.evaluate_player_availability import ResearchDixonColes

load_dotenv()
LOG = logging.getLogger("generate_lineup_forecast")

MODEL_VERSION = "dixon_coles_player_v1"
BASE_MODEL_VERSION = "dixon_coles_v2"
BASELINE_VERSION = "baseline_poisson_v1"
TIME_DECAY = 0.001
RIDGE = 1.0
MAX_GOALS = 9
MIN_TRAIN_MATCHES = 200
HALF_LIFE_DAYS = 180.0
SHRINKAGE_MINUTES = 450.0
MIN_POSITION_PRIOR_MINUTES = 450.0
TEAM_BASELINE_MATCHES = 3
TEAM_HISTORY_WINDOW = 5

LINEUP_STORE = Path(os.getenv("LINEUP_STORE", "data/lineups/confirmed_lineups.jsonl"))
FORECAST_STORE = Path(os.getenv("LINEUP_FORECAST_STORE", "data/lineups/lineup_forecasts.jsonl"))
PLAYER_HISTORY_PATH = Path(os.getenv("PLAYER_HISTORY_PATH", "fpl_complete_fixture_history.csv"))
PLAYER_FEATURES_PATH = Path(os.getenv("PLAYER_FEATURES_PATH", "data/processed/player_lineup_features.csv"))
PLAYER_MODEL_DATASET_PATH = Path(os.getenv("PLAYER_MODEL_DATASET_PATH", "data/processed/player_model_dataset.csv"))
PLAYER_MODEL_METADATA_PATH = Path(os.getenv("PLAYER_MODEL_METADATA_PATH", "reports/player_availability_backtest_metadata.csv"))

TEAM_ALIASES = {
    "afc bournemouth": ["bournemouth", "afc bournemouth"],
    "brighton and hove albion": ["brighton", "brighton and hove albion", "brighton hove albion"],
    "manchester city": ["man city", "manchester city"],
    "manchester united": ["man utd", "man united", "manchester united"],
    "newcastle united": ["newcastle", "newcastle united"],
    "nottingham forest": ["nottingham forest", "nottm forest", "nott'm forest", "forest"],
    "tottenham hotspur": ["tottenham", "tottenham hotspur", "spurs"],
    "west ham united": ["west ham", "west ham united"],
    "wolverhampton wanderers": ["wolves", "wolverhampton", "wolverhampton wanderers"],
    "ipswich town": ["ipswich", "ipswich town"],
    "coventry city": ["coventry", "coventry city"],
    "hull city": ["hull", "hull city"],
    "leeds united": ["leeds", "leeds united"],
    "leicester city": ["leicester", "leicester city"],
    "sunderland afc": ["sunderland", "sunderland afc"],
}

UPCOMING_SQL = text("""
    SELECT m.id AS match_id, m.season, m.matchweek, m.kickoff_utc, m.status,
           m.home_team_id, m.away_team_id, ht.name AS home_team, awt.name AS away_team
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams awt ON awt.id = m.away_team_id
    WHERE m.status = ANY(:statuses)
      AND m.home_goals IS NULL AND m.away_goals IS NULL
      AND m.kickoff_utc > NOW()
    ORDER BY m.kickoff_utc, m.id
    LIMIT :limit
""")

ONE_FIXTURE_SQL = text("""
    SELECT m.id AS match_id, m.season, m.matchweek, m.kickoff_utc, m.status,
           m.home_team_id, m.away_team_id, ht.name AS home_team, awt.name AS away_team
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams awt ON awt.id = m.away_team_id
    WHERE m.id = :match_id
""")

FINISHED_SQL = text("""
    SELECT id AS match_id, season, kickoff_utc, home_team_id, away_team_id,
           home_goals, away_goals
    FROM matches
    WHERE status = 'FINISHED'
      AND home_goals IS NOT NULL AND away_goals IS NOT NULL
      AND kickoff_utc < :date_cutoff
    ORDER BY kickoff_utc, id
""")


def normalise_season(value: Any) -> str:
    s = str(value).strip()
    if len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit():
        return f"{s[:4]}-{int(s[:4][:2]) * 100 + int(s[5:]):04d}"
    return s


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).casefold())


def canonical_team(value: Any) -> str:
    key = normalise_name(value)
    for canonical, aliases in TEAM_ALIASES.items():
        if key in {normalise_name(canonical), *(normalise_name(x) for x in aliases)}:
            return normalise_name(canonical)
    # Drop common terminal abbreviations when no configured alias applies.
    key = re.sub(r"(footballclub|fc|afc)$", "", key)
    return key


def player_identity(code: Any, name: Any) -> str:
    if code is not None and not pd.isna(code) and str(code).strip() not in {"", "nan", "None"}:
        try:
            return f"code:{int(float(code))}"
        except (TypeError, ValueError, OverflowError):
            return f"code:{str(code).strip().casefold()}"
    normalized = normalise_name(name)
    return f"name:{normalized}" if normalized else ""


def record_lineup_type(record: dict[str, Any]) -> str:
    value = record.get("lineup_type")
    if value in {"projected", "official_lineup"}:
        return value
    # Backwards compatibility with the original app, which only saved a
    # checkbox asserting the starting XI was checked against official team sheets.
    return "official_lineup" if record.get("confirmation_check") is True else "projected"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_num}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def latest_snapshot(records: list[dict[str, Any]], match_id: int,
                    lineup_type: str) -> dict[str, Any] | None:
    candidates = [
        r for r in records
        if int(r.get("match_id", -1)) == int(match_id)
        and record_lineup_type(r) == lineup_type
    ]
    return max(candidates, key=lambda r: int(r.get("revision", 0))) if candidates else None


def canonical_hash(payload: dict[str, Any]) -> str:
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def validate_xi(snapshot: dict[str, Any], side: str) -> list[dict[str, Any]]:
    lineup = snapshot.get(f"{side}_lineup")
    if not isinstance(lineup, list) or len(lineup) != 11:
        raise ValueError(f"Saved {side} lineup must contain exactly 11 players")
    ids = [str(p.get("element_id")) for p in lineup]
    if len(set(ids)) != 11:
        raise ValueError(f"Saved {side} lineup contains duplicate player IDs")
    valid_positions = {"GKP", "DEF", "MID", "FWD"}
    for player in lineup:
        if not isinstance(player, dict) or player.get("position") not in valid_positions:
            raise ValueError(f"Saved {side} lineup has a player with missing/invalid position")
    if sum(p.get("position") == "GKP" for p in lineup) != 1:
        raise ValueError(f"Saved {side} lineup must contain exactly one goalkeeper")
    return lineup


def load_player_history(path: Path, target_date: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Player history not found: {path}. Expected fpl_complete_fixture_history.csv."
        )
    df = pd.read_csv(path, low_memory=False)
    required = {"player_code", "player_name", "position", "kickoff_utc", "minutes", "xG", "xA"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Player history is missing columns: {missing}")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df["match_date_utc"] = df["kickoff_utc"].dt.floor("D")
    for col in ("player_code", "minutes", "xG", "xA"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["position"] = df["position"].map(lambda v: clean_text(v).upper())
    df["xgi"] = pd.to_numeric(df["xG"], errors="coerce") + pd.to_numeric(df["xA"], errors="coerce")
    # Strict cutoff: exclude every appearance on the target fixture's UTC date.
    df = df[df["match_date_utc"].notna() & (df["match_date_utc"] < target_date)].copy()
    df = df[df["minutes"].gt(0) & np.isfinite(df["xgi"])].copy()
    df["xgi"] = df["xgi"].clip(lower=0.0)
    return df


def build_weighted_rates(history: pd.DataFrame, as_of: pd.Timestamp) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    if history.empty:
        raise ValueError("No prior player appearance rows are available before the target date")
    age_days = ((as_of - history["match_date_utc"]).dt.total_seconds() / 86400.0).clip(lower=0).to_numpy()
    weights = np.exp(-math.log(2.0) * age_days / HALF_LIFE_DAYS)
    history = history.copy()
    history["weight"] = weights
    history["weighted_xgi"] = history["xgi"].to_numpy(dtype=float) * weights
    history["weighted_minutes"] = history["minutes"].to_numpy(dtype=float) * weights
    position_state: dict[str, dict[str, float]] = {}
    for position, group in history[history["position"].isin({"GKP", "DEF", "MID", "FWD"})].groupby("position"):
        xgi = float(group["weighted_xgi"].sum())
        minutes = float(group["weighted_minutes"].sum())
        position_state[str(position)] = {"xgi": xgi, "minutes": minutes}
    player_state: dict[str, dict[str, float]] = {}
    history["identity"] = [player_identity(code, name) for code, name in zip(history["player_code"], history["player_name"])]
    for identity, group in history[history["identity"].ne("")].groupby("identity"):
        player_state[identity] = {
            "xgi": float(group["weighted_xgi"].sum()),
            "minutes": float(group["weighted_minutes"].sum()),
        }
    return player_state, position_state


def rate_for_player(player: dict[str, Any], player_state: dict[str, dict[str, float]],
                    position_state: dict[str, dict[str, float]]) -> tuple[float | None, bool]:
    position = clean_text(player.get("position")).upper()
    pos = position_state.get(position)
    if not pos or pos["minutes"] < MIN_POSITION_PRIOR_MINUTES or pos["minutes"] <= 0:
        return None, False
    position_rate = 90.0 * pos["xgi"] / pos["minutes"]
    identity = player_identity(player.get("player_code"), player.get("player_name"))
    state = player_state.get(identity, {"xgi": 0.0, "minutes": 0.0})
    minutes = float(state["minutes"])
    denominator = minutes + SHRINKAGE_MINUTES
    rate = position_rate if denominator <= 0 else 90.0 * (
        float(state["xgi"]) + position_rate * SHRINKAGE_MINUTES / 90.0
    ) / denominator
    return max(0.0, float(rate)), minutes > 0


def team_recent_mean(features: pd.DataFrame, team_name: str,
                     target_date: pd.Timestamp) -> tuple[float | None, int]:
    team = canonical_team(team_name)
    prior = features[features["match_date_utc"] < target_date].copy()
    found: list[tuple[pd.Timestamp, float]] = []
    for row in prior.itertuples(index=False):
        row_home = canonical_team(getattr(row, "home_team", ""))
        row_away = canonical_team(getattr(row, "away_team", ""))
        if row_home == team:
            value = getattr(row, "home_xi_attack_xgi_per90_sum", np.nan)
        elif row_away == team:
            value = getattr(row, "away_xi_attack_xgi_per90_sum", np.nan)
        else:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            stamp = pd.Timestamp(getattr(row, "kickoff_utc"))
            found.append((stamp, numeric))
    found.sort(key=lambda pair: pair[0])
    values = [value for _, value in found[-TEAM_HISTORY_WINDOW:]]
    if len(values) < TEAM_BASELINE_MATCHES:
        return None, len(values)
    return float(np.mean(values)), len(values)


def load_features(path: Path, target_date: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Player lineup features not found: {path}. Rebuild them with --seasons 2022-23 2023-24 2024-25 2025-26 2026-27."
        )
    df = pd.read_csv(path, low_memory=False)
    required = {"kickoff_utc", "match_date_utc", "home_team", "away_team",
                "home_xi_attack_xgi_per90_sum", "away_xi_attack_xgi_per90_sum"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Player feature file is missing columns: {missing}")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df["match_date_utc"] = pd.to_datetime(df["match_date_utc"], utc=True, errors="coerce").dt.floor("D")
    for col in ("home_xi_attack_xgi_per90_sum", "away_xi_attack_xgi_per90_sum"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df["match_date_utc"].notna() & (df["match_date_utc"] < target_date)
    has_any_lineup = (
        np.isfinite(df["home_xi_attack_xgi_per90_sum"])
        | np.isfinite(df["away_xi_attack_xgi_per90_sum"])
    )
    valid_dates = df.loc[valid & has_any_lineup, "match_date_utc"]
    if valid_dates.empty:
        raise ValueError("No valid historical lineup-strength rows exist before the target fixture date")
    most_recent = valid_dates.max()
    days_old = (target_date - most_recent).total_seconds() / 86400.0
    if days_old > 370:
        raise ValueError(
            f"Latest usable lineup-strength history ends on {most_recent.date()} (about {days_old:.0f} days before this fixture). "
            "Rebuild it to include 2025-26 and 2026-27: python -m src.build_player_lineup_features --seasons 2022-23 2023-24 2024-25 2025-26 2026-27"
        )
    return df


def load_frozen_player_params(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Player model metadata not found: {path}. Run the full player-availability backtest first."
        )
    df = pd.read_csv(path)
    required = {"theta", "feature_scaler_mean", "feature_scaler_std"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Player-model metadata is missing columns: {missing}")
    row = df.iloc[-1]
    params = {key: float(row[key]) for key in required}
    if not all(np.isfinite(v) for v in params.values()) or params["feature_scaler_std"] <= 0:
        raise ValueError(f"Invalid frozen player-model parameters: {params}")
    return params


def load_upcoming_fixture(engine, match_id: int | None, limit: int = 50) -> pd.DataFrame:
    with engine.connect() as connection:
        if match_id is None:
            return pd.read_sql_query(UPCOMING_SQL, connection,
                                     params={"statuses": ["SCHEDULED", "TIMED"], "limit": limit})
        row = pd.read_sql_query(ONE_FIXTURE_SQL, connection, params={"match_id": int(match_id)})
    if row.empty:
        raise ValueError(f"No fixture with match_id={match_id} exists in the database")
    row["kickoff_utc"] = pd.to_datetime(row["kickoff_utc"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    if row.iloc[0]["status"] not in {"SCHEDULED", "TIMED"}:
        raise ValueError(f"Fixture {match_id} is not upcoming (status={row.iloc[0]['status']})")
    if pd.isna(row.iloc[0]["kickoff_utc"]) or row.iloc[0]["kickoff_utc"] <= now:
        raise ValueError(f"Fixture {match_id} has already kicked off or has invalid kickoff time")
    return row


def load_finished_training(engine, target_date: pd.Timestamp, model_dataset_path: Path,
                          frozen: dict[str, float]) -> pd.DataFrame:
    with engine.connect() as connection:
        train = pd.read_sql_query(FINISHED_SQL, connection,
                                  params={"date_cutoff": target_date.to_pydatetime()})
    if train.empty or len(train) < MIN_TRAIN_MATCHES:
        raise ValueError(f"Need at least {MIN_TRAIN_MATCHES} finished matches strictly before target UTC date; got {len(train)}")
    train["kickoff_utc"] = pd.to_datetime(train["kickoff_utc"], utc=True, errors="coerce")
    train = train.dropna(subset=["kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals"]).copy()
    if train.empty or len(train) < MIN_TRAIN_MATCHES:
        raise ValueError(f"Only {len(train)} valid completed matches remain before target date")
    train["home_team_id"] = train["home_team_id"].astype(int)
    train["away_team_id"] = train["away_team_id"].astype(int)
    train["home_goals"] = train["home_goals"].astype(int)
    train["away_goals"] = train["away_goals"].astype(int)

    if not model_dataset_path.is_file():
        raise FileNotFoundError(
            f"Joined player model dataset not found: {model_dataset_path}. First rebuild lineup features for 2022-23 through 2026-27, "
            "then run python -m src.prepare_player_model_dataset."
        )
    features = pd.read_csv(model_dataset_path, low_memory=False)
    required = {"match_id", "home_xi_attack_delta_vs_recent_team_mean",
                "away_xi_attack_delta_vs_recent_team_mean", "lineup_delta_features_ready"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Joined player dataset is missing columns: {missing}")
    features["match_id"] = pd.to_numeric(features["match_id"], errors="coerce")
    features = features.dropna(subset=["match_id"]).copy()
    features["match_id"] = features["match_id"].astype(int)
    if features["match_id"].duplicated().any():
        raise ValueError("Joined player dataset contains duplicate database match_id values")
    for col in ("home_xi_attack_delta_vs_recent_team_mean", "away_xi_attack_delta_vs_recent_team_mean"):
        features[col] = pd.to_numeric(features[col], errors="coerce")
    flags = features["lineup_delta_features_ready"].astype(str).str.strip().str.casefold().isin({"true", "1", "yes", "y"})
    flags &= np.isfinite(features["home_xi_attack_delta_vs_recent_team_mean"])
    flags &= np.isfinite(features["away_xi_attack_delta_vs_recent_team_mean"])
    features["lineup_delta_features_ready"] = flags.to_numpy(dtype=bool)
    features = features[["match_id", "home_xi_attack_delta_vs_recent_team_mean",
                         "away_xi_attack_delta_vs_recent_team_mean", "lineup_delta_features_ready"]].copy()
    train = train.merge(features, how="left", on="match_id", validate="one_to_one")
    train["player_x_home_z"] = 0.0
    train["player_x_away_z"] = 0.0
    ready = (
        train["lineup_delta_features_ready"].fillna(False).astype(bool)
        & train["home_xi_attack_delta_vs_recent_team_mean"].notna()
        & train["away_xi_attack_delta_vs_recent_team_mean"].notna()
        & np.isfinite(train["home_xi_attack_delta_vs_recent_team_mean"])
        & np.isfinite(train["away_xi_attack_delta_vs_recent_team_mean"])
    )
    # As in the historical experiment, unavailable historical lineup features
    # are treated as neutral (z=0). Ready features use the frozen 2022-23 scaler.
    train.loc[ready, "player_x_home_z"] = (
        train.loc[ready, "home_xi_attack_delta_vs_recent_team_mean"] - frozen["feature_scaler_mean"]
    ) / frozen["feature_scaler_std"]
    train.loc[ready, "player_x_away_z"] = (
        train.loc[ready, "away_xi_attack_delta_vs_recent_team_mean"] - frozen["feature_scaler_mean"]
    ) / frozen["feature_scaler_std"]
    return train


def league_average_rates(train: pd.DataFrame, reference_date: pd.Timestamp) -> tuple[float, float]:
    kickoff = pd.to_datetime(train["kickoff_utc"], utc=True)
    age = ((reference_date - kickoff).dt.total_seconds() / 86400.0).clip(lower=0).to_numpy()
    weights = np.exp(-TIME_DECAY * age)
    home = float(np.average(train["home_goals"].to_numpy(dtype=float), weights=weights))
    away = float(np.average(train["away_goals"].to_numpy(dtype=float), weights=weights))
    if not np.isfinite([home, away]).all() or home <= 0 or away <= 0:
        raise ValueError(f"Invalid league-average rates home={home}, away={away}")
    return home, away


def validate_distribution(matrix: np.ndarray, probs: tuple[float, float, float]) -> None:
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(probs, dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("Forecast contains non-finite probabilities")
    if (matrix < 0).any() or (vector < 0).any() or (vector > 1).any():
        raise ValueError("Forecast probabilities are outside valid bounds")
    if abs(float(matrix.sum()) - 1.0) > 1e-8 or abs(float(vector.sum()) - 1.0) > 1e-8:
        raise ValueError("Forecast probabilities do not sum to one")
    derived = np.asarray(outcome_probs(matrix), dtype=float)
    if np.max(np.abs(derived - vector)) > 1e-8:
        raise ValueError("1X2 probabilities disagree with the scoreline matrix")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_forecast(engine, match_id: int, lineup_type: str,
                  dataset_path: Path, features_path: Path, model_dataset_path: Path,
                  metadata_path: Path, lineup_path: Path) -> dict[str, Any]:
    now = pd.Timestamp.now(tz="UTC")
    snapshots = load_jsonl(lineup_path)
    snapshot = latest_snapshot(snapshots, match_id, lineup_type)
    if snapshot is None:
        raise ValueError(
            f"No saved {lineup_type} lineup snapshot found for match_id={match_id} in {lineup_path}. "
            "Run the lineup entry app first."
        )
    home_xi = validate_xi(snapshot, "home")
    away_xi = validate_xi(snapshot, "away")

    fixture = load_upcoming_fixture(engine, match_id).iloc[0]
    kickoff = pd.Timestamp(fixture["kickoff_utc"]).tz_convert("UTC")
    target_date = kickoff.floor("D")
    created_at = pd.Timestamp.now(tz="UTC")
    submitted = pd.to_datetime(snapshot.get("confirmed_at_utc"), utc=True, errors="coerce")
    if pd.isna(submitted):
        raise ValueError("Lineup snapshot has no valid confirmed_at_utc timestamp")
    if submitted >= kickoff:
        raise ValueError("Lineup snapshot was saved at/after kickoff; refusing to create a pre-match forecast")
    if created_at >= kickoff:
        raise ValueError("Current time is at/after kickoff; refusing to create a pre-match forecast")
    if snapshot.get("lineup_type") and snapshot.get("lineup_type") != lineup_type:
        raise ValueError("Selected lineup type does not match the saved snapshot type")
    if int(snapshot.get("match_id", -1)) != int(match_id):
        raise ValueError("Lineup snapshot match_id mismatch")
    if canonical_team(snapshot.get("home_team", "")) != canonical_team(fixture["home_team"]):
        raise ValueError("Saved lineup home team does not match the database fixture")
    if canonical_team(snapshot.get("away_team", "")) != canonical_team(fixture["away_team"]):
        raise ValueError("Saved lineup away team does not match the database fixture")

    history = load_player_history(dataset_path, target_date)
    player_state, position_state = build_weighted_rates(history, target_date)
    feature_history = load_features(features_path, target_date)
    frozen = load_frozen_player_params(metadata_path)

    def side_features(team_name: str, lineup: list[dict[str, Any]]) -> dict[str, Any]:
        rates = []
        prior_players = 0
        rate_missing_players = []
        for player in lineup:
            rate, had_history = rate_for_player(player, player_state, position_state)
            if rate is None or not np.isfinite(rate):
                rate_missing_players.append(player.get("player_name", "unknown"))
            else:
                rates.append(float(rate))
            prior_players += int(had_history)
        current_total = float(sum(rates)) if len(rates) == 11 and not rate_missing_players else None
        recent_mean, prior_team_lineups = team_recent_mean(feature_history, team_name, target_date)
        raw_delta = current_total - recent_mean if current_total is not None and recent_mean is not None else None
        if raw_delta is None:
            z = 0.0  # neutral-centred fallback; explicitly marked below
            status = "neutral_center_fallback_missing_player_or_team_history"
        else:
            z = (raw_delta - frozen["feature_scaler_mean"]) / frozen["feature_scaler_std"]
            status = "lineup_delta_available"
        return {
            "team": team_name,
            "starting_xi_player_count": len(lineup),
            "players_with_prior_individual_history": prior_players,
            "estimated_starting_xi_xgi_per90_sum": current_total,
            "recent_team_lineup_mean_xgi_per90_sum": recent_mean,
            "lineup_delta_raw": raw_delta,
            "lineup_delta_standardized": float(z),
            "prior_team_lineups_used": prior_team_lineups,
            "player_rate_missing_names": rate_missing_players,
            "feature_status": status,
        }

    home_feature = side_features(str(fixture["home_team"]), home_xi)
    away_feature = side_features(str(fixture["away_team"]), away_xi)

    frozen = load_frozen_player_params(metadata_path)
    train = load_finished_training(engine, target_date, model_dataset_path, frozen)
    reference_date = target_date
    train_cutoff = pd.to_datetime(train["kickoff_utc"], utc=True).max()
    base_home, base_away = league_average_rates(train, reference_date)
    base_matrix = score_matrix(base_home, base_away, 0.0, MAX_GOALS)
    base_probs = tuple(map(float, outcome_probs(base_matrix)))

    # Control is the current ordinary Dixon-Coles estimator.
    control_model = DixonColesModel(time_decay=TIME_DECAY, ridge=RIDGE)
    control_model.fit(train, reference_date=reference_date)

    # The experimental model uses the same research fitter and frozen theta as
    # the historical evaluation, with historical lineup features in its fit.
    player_model = ResearchDixonColes(ridge=RIDGE, theta_ridge=2.0)
    player_model.fit(train, reference_date=reference_date, fixed_theta=frozen["theta"])

    cold_start_reason = None
    try:
        control_prediction = control_model.predict_probs(
            int(fixture["home_team_id"]), int(fixture["away_team_id"]), MAX_GOALS
        )
    except UnknownTeamError as exc:
        control_prediction = None
        cold_start_reason = f"unknown_team_id:{exc}"
    player_prediction = player_model.predict_probs(
        int(fixture["home_team_id"]), int(fixture["away_team_id"]),
        float(home_feature["lineup_delta_standardized"]),
        float(away_feature["lineup_delta_standardized"]), MAX_GOALS,
    )

    if control_prediction is None or player_prediction is None:
        # Use the same labelled neutral cold-start fallback for both branches;
        # do not claim a lineup adjustment was applied in this case.
        control_matrix = base_matrix
        control_probs = base_probs
        control_lam, control_mu = base_home, base_away
        player_matrix = base_matrix
        player_probs = base_probs
        player_lam, player_mu = base_home, base_away
        prediction_source = "league_average_cold_start_no_player_adjustment"
        control_version = "dixon_coles_v2_cold_start_baseline_v1"
        player_version = "dixon_coles_player_v1_cold_start_baseline_v1"
        player_adjustment_applied = False
        if cold_start_reason is None:
            cold_start_reason = "team_not_in_player_model_team_index"
    else:
        c_home, c_draw, c_away, control_lam, control_mu, control_matrix = control_prediction
        control_probs = tuple(map(float, (c_home, c_draw, c_away)))
        player_probs, player_lam, player_mu, player_matrix = player_prediction
        player_probs = tuple(map(float, player_probs))
        player_lam, player_mu = float(player_lam), float(player_mu)
        prediction_source = "dixon_coles_with_frozen_player_lineup_adjustment"
        control_version = BASE_MODEL_VERSION
        player_version = MODEL_VERSION
        player_adjustment_applied = True

    validate_distribution(np.asarray(control_matrix, dtype=float), control_probs)
    validate_distribution(np.asarray(player_matrix, dtype=float), player_probs)

    record = {
        "schema_version": 2,
        "forecast_model_version": player_version,
        "requested_model_version": MODEL_VERSION,
        "base_model_version": control_version,
        "prediction_source": prediction_source,
        "match_id": int(match_id),
        "season": str(fixture["season"]),
        "matchweek": int(fixture["matchweek"]) if pd.notna(fixture["matchweek"]) else None,
        "kickoff_utc": kickoff.isoformat(),
        "home_team": str(fixture["home_team"]),
        "away_team": str(fixture["away_team"]),
        "lineup_type": lineup_type,
        "lineup_revision": int(snapshot.get("revision", 0)),
        "lineup_saved_at_utc": submitted.isoformat(),
        "forecast_generated_at_utc": created_at.isoformat(),
        "training_matches": int(len(train)),
        "training_cutoff_utc": train_cutoff.isoformat(),
        "training_max_goal_date": str(train_cutoff.date()),
        "theta_frozen": float(frozen["theta"]),
        "feature_scaler_mean_frozen": float(frozen["feature_scaler_mean"]),
        "feature_scaler_std_frozen": float(frozen["feature_scaler_std"]),
        "player_adjustment_applied": bool(player_adjustment_applied),
        "cold_start_reason": cold_start_reason,
        "home_lineup_feature": home_feature,
        "away_lineup_feature": away_feature,
        "control_forecast": {
            "model_version": control_version,
            "probabilities": {"home_win": float(control_probs[0]), "draw": float(control_probs[1]), "away_win": float(control_probs[2])},
            "expected_goals": {"home": float(control_lam), "away": float(control_mu)},
            "scoreline_matrix": np.asarray(control_matrix, dtype=float).tolist(),
        },
        "player_forecast": {
            "model_version": player_version,
            "probabilities": {"home_win": float(player_probs[0]), "draw": float(player_probs[1]), "away_win": float(player_probs[2])},
            "expected_goals": {"home": float(player_lam), "away": float(player_mu)},
            "scoreline_matrix": np.asarray(player_matrix, dtype=float).tolist(),
        },
        "training_note": "Only FINISHED matches strictly before the target UTC calendar date were used.",
        "research_note": "Player-aware forecast is experimental; historical held-out gains were small and confidence intervals included zero.",
    }
    # Exclude generation time from identity so identical inputs are idempotent.
    hash_payload = {key: value for key, value in record.items()
                    if key not in {"forecast_generated_at_utc", "forecast_hash"}}
    record["forecast_hash"] = canonical_hash(hash_payload)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", type=int, help="database match ID; use --list-upcoming to find IDs")
    parser.add_argument("--lineup-type", choices=["projected", "official_lineup"],
                        help="which saved lineup snapshot to use")
    parser.add_argument("--list-upcoming", action="store_true", help="list future fixtures with IDs and exit")
    parser.add_argument("--limit", type=int, default=50, help="fixture count for --list-upcoming")
    parser.add_argument("--history", default=str(PLAYER_HISTORY_PATH), help="FPL player history CSV")
    parser.add_argument("--features", default=str(PLAYER_FEATURES_PATH), help="historical lineup features CSV")
    parser.add_argument("--model-dataset", default=str(PLAYER_MODEL_DATASET_PATH), help="joined historical match/player feature dataset")
    parser.add_argument("--metadata", default=str(PLAYER_MODEL_METADATA_PATH), help="frozen player model metadata CSV")
    parser.add_argument("--lineup-store", default=str(LINEUP_STORE), help="append-only lineup snapshot JSONL")
    parser.add_argument("--forecast-store", default=str(FORECAST_STORE), help="append-only forecast JSONL")
    parser.add_argument("--save", action="store_true", help="append forecast to JSONL; default is preview only")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if not args.list_upcoming and args.match_id is None:
        parser.error("Supply --match-id, or use --list-upcoming")
    if args.match_id is not None and not args.lineup_type:
        parser.error("--lineup-type is required with --match-id (projected or official_lineup)")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine = get_sqlalchemy_engine()
    try:
        if args.list_upcoming:
            with engine.connect() as connection:
                upcoming = pd.read_sql_query(UPCOMING_SQL, connection,
                    params={"statuses": ["SCHEDULED", "TIMED"], "limit": args.limit})
            if upcoming.empty:
                print("No upcoming fixtures found.")
                return 0
            upcoming["kickoff_utc"] = pd.to_datetime(upcoming["kickoff_utc"], utc=True, errors="coerce")
            print("UPCOMING FIXTURES (UTC)")
            print(upcoming[["match_id", "kickoff_utc", "season", "home_team", "away_team", "status"]].to_string(index=False))
            return 0

        record = make_forecast(
            engine, int(args.match_id), str(args.lineup_type), Path(args.history), Path(args.features),
            Path(args.model_dataset), Path(args.metadata), Path(args.lineup_store),
        )
        forecast_path = Path(args.forecast_store)
        if args.save:
            existing = load_jsonl(forecast_path)
            duplicate = next((row for row in existing if row.get("forecast_hash") == record["forecast_hash"]), None)
            if duplicate:
                print("This exact forecast already exists; no duplicate was appended.")
                record = duplicate
            else:
                append_jsonl(forecast_path, record)
                print(f"Saved experimental forecast to {forecast_path}")
        else:
            print("DRY RUN: forecast was not saved. Re-run with --save to append it to the experimental forecast log.")

        print("\nLINEUP-AWARE FORECAST")
        print(f"Fixture: {record['home_team']} vs {record['away_team']} (match_id={record['match_id']})")
        print(f"Lineup type/revision: {record['lineup_type']} / {record['lineup_revision']}")
        print(f"Kickoff UTC: {record['kickoff_utc']}")
        print(f"Model recorded: {record['forecast_model_version']} (base={record['base_model_version']})")
        print(f"Prediction source: {record['prediction_source']}")
        print(f"Training matches: {record['training_matches']}; cutoff={record['training_cutoff_utc']}")
        print(f"Player adjustment applied: {record['player_adjustment_applied']}")
        for branch_key in ("control_forecast", "player_forecast"):
            branch = record[branch_key]
            print(f"\n{branch_key.replace('_', ' ').title()} ({branch['model_version']})")
            print("  Probabilities: home={home_win:.4f} draw={draw:.4f} away={away_win:.4f}".format(**branch["probabilities"]))
            print("  Expected goals: home={home:.3f} away={away:.3f}".format(**branch["expected_goals"]))
        for side in ("home", "away"):
            feat = record[f"{side}_lineup_feature"]
            print(f"{side.title()} XI feature: status={feat['feature_status']}, raw_delta={feat['lineup_delta_raw']}, z={feat['lineup_delta_standardized']:.4f}, team_lineups_used={feat['prior_team_lineups_used']}")
        if record.get("cold_start_reason"):
            print(f"Cold-start reason: {record['cold_start_reason']}")
        print(f"Forecast hash: {record['forecast_hash']}")
        return 0
    except Exception as exc:
        LOG.exception("Lineup-aware forecast failed: %s", exc)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
