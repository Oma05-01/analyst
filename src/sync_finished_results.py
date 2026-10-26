#!/usr/bin/env python3
"""Sync finished current-season Premier League results from football-data.org.

Safe by default: reads the API and PostgreSQL and prints a precheck only.
Use --apply to write match results. This script never inserts, updates, or deletes
rows in predictions, prediction_scores, or match_previews.

Run from the repository root:
    python -m src.sync_finished_results
    python -m src.sync_finished_results --apply
    python -m src.sync_finished_results --season-start-year 2026

The default season is inferred from the current date (July onward => current year;
January-June => previous year). Override it when testing or syncing another season.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from src.db import get_db_connection

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
REQUEST_TIMEOUT = 20
UTC = timezone.utc

# Conservative aliases for source labels that differ from canonical database names.
# Aliases only resolve names; they never create a team or fuzzy-match a club.
SOURCE_TEAM_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "newcastle": "newcastle united",
    "nottm forest": "nottingham forest",
    "notts forest": "nottingham forest",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "west ham": "west ham united",
    "brighton": "brighton and hove albion",
    "leeds": "leeds united",
    "luton": "luton town",
    "ipswich": "ipswich town",
    "sheffield utd": "sheffield united",
    "sheffield weds": "sheffield wednesday",
    "west brom": "west bromwich albion",
    "wba": "west bromwich albion",
    "norwich": "norwich city",
    "palace": "crystal palace",
    "bournemouth": "bournemouth",
    "coventry": "coventry city",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def team_key(value: object) -> str:
    if value is None:
        return ""
    name = str(value).strip().casefold()
    if name in {"", "none", "nan"}:
        return ""
    name = name.replace("&", " and ").replace("'", "").replace(".", "")
    name = re.sub(r"^afc\s+", "", name)
    name = re.sub(r"\s+fc$", "", name)
    name = re.sub(r"\s+afc$", "", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def inferred_season_start_year(now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    return now.year if now.month >= 7 else now.year - 1


def load_team_mapping(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, aliases FROM teams")
        rows = cur.fetchall()

    mapping: dict[str, int] = {}

    def register(label: object, team_id: int) -> None:
        key = team_key(label)
        if not key:
            return
        old = mapping.get(key)
        if old is not None and old != int(team_id):
            raise ValueError(f"Ambiguous database team alias {label!r}: IDs {old} and {team_id}")
        mapping[key] = int(team_id)

    for team_id, name, aliases in rows:
        register(name, team_id)
        if aliases:
            parsed = json.loads(aliases) if isinstance(aliases, str) else aliases
            if isinstance(parsed, (list, tuple)):
                for alias in parsed:
                    register(alias, team_id)
            elif isinstance(parsed, dict):
                for alias in parsed.keys():
                    register(alias, team_id)
            else:
                raise ValueError(f"Unsupported aliases for team {name!r}: {type(parsed).__name__}")

    return mapping


def resolve_team(raw_name: object, team_map: dict[str, int]) -> int | None:
    key = team_key(raw_name)
    candidates = [key]
    canonical = SOURCE_TEAM_ALIASES.get(key)
    if canonical:
        candidates.append(team_key(canonical))
    ids = {team_map[candidate] for candidate in candidates if candidate in team_map}
    if len(ids) > 1:
        raise ValueError(f"Source team {raw_name!r} maps to multiple database team IDs: {sorted(ids)}")
    return next(iter(ids)) if ids else None


def parse_nonnegative_int(value: object, label: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Missing {label}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if number < 0 or str(number) != str(value).strip():
        # JSON numeric values may arrive as 2 or 2.0; accept integral floats.
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {label}: {value!r}")
        if not float_value.is_integer() or float_value < 0:
            raise ValueError(f"Invalid {label}: {value!r}")
        number = int(float_value)
    return number


def parse_api_matches(matches: list[dict], season_start_year: int) -> list[dict]:
    expected_season = f"{season_start_year}-{season_start_year + 1}"
    records: list[dict] = []
    seen_keys = set()

    for raw in matches:
        status = str(raw.get("status") or "").upper()
        if status != "FINISHED":
            # This endpoint is filtered, but enforce it locally too.
            continue
        competition = (raw.get("competition") or {}).get("code")
        if competition and competition != "PL":
            raise ValueError(f"Unexpected competition {competition!r} in result payload")

        season_info = raw.get("season") or {}
        start_date = season_info.get("startDate")
        if not start_date or not str(start_date)[:4].isdigit():
            raise ValueError(f"Match {raw.get('id')} has missing/invalid season.startDate")
        season_year = int(str(start_date)[:4])
        season = f"{season_year}-{season_year + 1}"
        if season != expected_season:
            raise ValueError(f"API returned {season} fixture while requesting {expected_season}")

        kickoff_text = raw.get("utcDate")
        if not kickoff_text:
            raise ValueError(f"Match {raw.get('id')} has no utcDate")
        try:
            kickoff = datetime.fromisoformat(str(kickoff_text).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Match {raw.get('id')} has invalid utcDate {kickoff_text!r}") from exc
        if kickoff.tzinfo is None:
            raise ValueError(f"Match {raw.get('id')} utcDate lacks timezone information")
        kickoff = kickoff.astimezone(UTC)

        home = (raw.get("homeTeam") or {}).get("name")
        away = (raw.get("awayTeam") or {}).get("name")
        if not home or not away:
            raise ValueError(f"Match {raw.get('id')} has missing home/away team name")

        score = raw.get("score") or {}
        full_time = score.get("fullTime") or {}
        home_goals = parse_nonnegative_int(full_time.get("home"), f"match {raw.get('id')} fullTime.home")
        away_goals = parse_nonnegative_int(full_time.get("away"), f"match {raw.get('id')} fullTime.away")

        matchday = raw.get("matchday")
        if matchday is not None:
            matchday = parse_nonnegative_int(matchday, f"match {raw.get('id')} matchday")
            if matchday == 0:
                raise ValueError(f"Match {raw.get('id')} has invalid matchday 0")

        record = {
            "api_match_id": raw.get("id"),
            "season": season,
            "matchweek": matchday,
            "kickoff_utc": kickoff,
            "home_name": str(home).strip(),
            "away_name": str(away).strip(),
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
        records.append(record)

    if not records:
        raise ValueError(
            f"The API returned no completed matches for {expected_season}. "
            "Refusing to treat an empty response as a successful sync."
        )
    return records


def fetch_finished_matches(season_start_year: int) -> list[dict]:
    if not API_KEY:
        raise RuntimeError("FOOTBALL_DATA_API_KEY is not set in the environment/.env")
    url = BASE_URL
    params = {"status": "FINISHED", "season": season_start_year}
    headers = {"X-Auth-Token": API_KEY, "User-Agent": "football-analyst-agent/1.0"}
    logging.info("Fetching finished Premier League matches for season starting %s", season_start_year)
    try:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        body = getattr(getattr(exc, "response", None), "text", "")
        detail = f"; response body: {body[:500]}" if body else ""
        raise RuntimeError(f"Football-data.org request failed: {exc}{detail}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Football-data.org returned a non-JSON response") from exc
    matches = data.get("matches")
    if not isinstance(matches, list):
        raise RuntimeError("API response did not include a matches list")
    return parse_api_matches(matches, season_start_year)


def fetch_db_fixtures(conn) -> dict[tuple[str, int, int], dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.season, m.home_team_id, m.away_team_id,
                   m.matchweek, m.kickoff_utc, m.status, m.home_goals, m.away_goals,
                   EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.id) AS has_predictions
            FROM matches m
            """
        )
        rows = cur.fetchall()
    fixtures: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        record = {
            "id": int(row[0]), "season": str(row[1]),
            "home_team_id": int(row[2]), "away_team_id": int(row[3]),
            "matchweek": row[4], "kickoff_utc": row[5], "status": row[6],
            "home_goals": row[7], "away_goals": row[8], "has_predictions": bool(row[9]),
        }
        key = (record["season"], record["home_team_id"], record["away_team_id"])
        if key in fixtures:
            raise ValueError(f"Duplicate fixture key in database for {key}")
        fixtures[key] = record
    return fixtures


def prepare_records(conn, records: list[dict]) -> tuple[list[dict], dict]:
    team_map = load_team_mapping(conn)
    db_fixtures = fetch_db_fixtures(conn)
    prepared = []
    unmapped = set()
    seen_keys = set()

    for rec in records:
        home_id = resolve_team(rec["home_name"], team_map)
        away_id = resolve_team(rec["away_name"], team_map)
        if home_id is None:
            unmapped.add(rec["home_name"])
        if away_id is None:
            unmapped.add(rec["away_name"])
        if home_id is None or away_id is None:
            continue
        if home_id == away_id:
            raise ValueError(f"Same team mapped as home and away for {rec['home_name']} vs {rec['away_name']}")

        key = (rec["season"], home_id, away_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate completed fixture from API: {key}")
        seen_keys.add(key)
        existing = db_fixtures.get(key)
        matchweek = rec["matchweek"]
        if matchweek is None and existing is not None:
            matchweek = existing["matchweek"]
        if matchweek is None:
            raise ValueError(
                f"No matchday for new fixture {rec['home_name']} vs {rec['away_name']}; "
                "matches.matchweek is NOT NULL, so no writes can be made."
            )
        prepared.append({
            **rec,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "matchweek": int(matchweek),
            "existing": existing,
        })

    if unmapped:
        raise ValueError(
            "These API club names could not be mapped uniquely to teams.id: "
            f"{sorted(unmapped)}. No writes were made. Add verified aliases to the teams table "
            "or SOURCE_TEAM_ALIASES, then rerun."
        )
    if len(prepared) != len(records):
        raise ValueError("Not every API result was prepared; no writes were made")
    return prepared, db_fixtures


def print_precheck(prepared: list[dict], season_start_year: int, apply: bool) -> None:
    season = f"{season_start_year}-{season_start_year + 1}"
    existing = [r for r in prepared if r["existing"] is not None]
    new = [r for r in prepared if r["existing"] is None]
    predicted_existing = [r for r in existing if r["existing"]["has_predictions"]]
    needing_result_change = [
        r for r in existing
        if str(r["existing"]["status"]).upper() != "FINISHED"
        or r["existing"]["home_goals"] != r["home_goals"]
        or r["existing"]["away_goals"] != r["away_goals"]
    ]
    dates = sorted(r["kickoff_utc"].date().isoformat() for r in prepared)
    logging.info("\nFINISHED-RESULTS SYNC PRECHECK")
    logging.info("Mode: %s", "APPLY (database changes will be committed)" if apply else "DRY RUN (no database writes)")
    logging.info("Season: %s", season)
    logging.info("Finished API results validated: %d", len(prepared))
    logging.info("Source kickoff-date span: %s to %s", dates[0], dates[-1])
    logging.info("Existing fixtures matched: %d", len(existing))
    logging.info("Existing fixtures needing status/score update: %d", len(needing_result_change))
    logging.info("Existing fixtures with prediction records: %d", len(predicted_existing))
    logging.info("New fixtures to insert: %d", len(new))
    logging.info("Prediction table writes: 0 (predictions, prediction_scores, match_previews untouched)")

    interesting = [r for r in prepared if "ipswich" in team_key(r["home_name"]) and "nottingham forest" in team_key(r["away_name"])]
    for r in interesting:
        ex = r["existing"]
        logging.info(
            "Ipswich vs Nottingham Forest check: API score %d-%d; local fixture %s; prediction-backed=%s",
            r["home_goals"], r["away_goals"],
            f"match_id={ex['id']} status={ex['status']} stored_score={ex['home_goals']}-{ex['away_goals']}"
            if ex else "not present (will insert)",
            bool(ex and ex["has_predictions"]),
        )

    if not apply:
        logging.info("Dry run complete. Review results, then rerun with --apply to commit the sync.")


def apply_results(conn, prepared: list[dict]) -> int:
    sql = """
        INSERT INTO matches AS current_match (
            season, matchweek, kickoff_utc, home_team_id, away_team_id,
            status, home_goals, away_goals
        )
        VALUES (%s, %s, %s, %s, %s, 'FINISHED', %s, %s)
        ON CONFLICT (season, home_team_id, away_team_id)
        DO UPDATE SET
            status = 'FINISHED',
            home_goals = EXCLUDED.home_goals,
            away_goals = EXCLUDED.away_goals,
            matchweek = CASE
                WHEN EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = current_match.id)
                    THEN current_match.matchweek
                ELSE EXCLUDED.matchweek
            END,
            kickoff_utc = CASE
                WHEN EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = current_match.id)
                    THEN current_match.kickoff_utc
                ELSE EXCLUDED.kickoff_utc
            END
    """
    count = 0
    try:
        with conn.cursor() as cur:
            for row in prepared:
                cur.execute(
                    sql,
                    (
                        row["season"], row["matchweek"], row["kickoff_utc"],
                        row["home_team_id"], row["away_team_id"],
                        row["home_goals"], row["away_goals"],
                    ),
                )
                count += 1
        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season-start-year", type=int, default=None,
        help="season start year (default: infer from current date, e.g. 2026 for 2026-2027)",
    )
    parser.add_argument("--apply", action="store_true", help="commit result updates (default is dry-run)")
    args = parser.parse_args()

    season_year = args.season_start_year or inferred_season_start_year()
    if season_year < 2000 or season_year > datetime.now(UTC).year + 1:
        parser.error("--season-start-year is outside the supported range")

    conn = None
    try:
        records = fetch_finished_matches(season_year)
        logging.info("API returned %d completed matches", len(records))
        conn = get_db_connection()
        prepared, _ = prepare_records(conn, records)
        print_precheck(prepared, season_year, args.apply)
        if args.apply:
            count = apply_results(conn, prepared)
            logging.info("Sync committed successfully. Result rows processed: %d", count)
            logging.info("Next: run `python -m src.score_predictions` to score eligible locked predictions.")
        return 0
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        logging.error("Sync aborted: %s. No changes committed by this script.", exc)
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
