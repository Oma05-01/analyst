import os
import json
import requests
import pandas as pd
from datetime import datetime
from db import get_db_connection
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
PL_MATCHES_URL = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"

def get_or_create_team(cursor, team_name, team_map):
    clean_name = str(team_name).strip()
    key = clean_name.lower()
    
    if key in team_map:
        return team_map[key]

    # Handle common suffixes
    base_name = clean_name
    for suffix in [" FC", " AFC"]:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)].strip()
            
    if base_name.lower() in team_map:
        return team_map[base_name.lower()]

    # Insert newly promoted team into database
    short_code = base_name[:3].upper()
    aliases = json.dumps([clean_name, base_name])
    cursor.execute(
        "INSERT INTO teams (name, short_code, aliases) VALUES (%s, %s, %s) RETURNING id;",
        (base_name, short_code, aliases)
    )
    new_id = cursor.fetchone()[0]
    team_map[key] = new_id
    team_map[base_name.lower()] = new_id
    print(f"Registered new team in database: '{base_name}' (ID: {new_id})")
    return new_id

def fetch_and_store_live_fixtures():
    if not API_KEY:
        print("Error: FOOTBALL_DATA_API_KEY environment variable not set.")
        return

    headers = {"X-Auth-Token": API_KEY}
    print("Fetching scheduled fixtures from football-data.org API...")

    try:
        response = requests.get(
            PL_MATCHES_URL,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch fixtures: {exc}")
        return

    data = response.json()
    matches = data.get("matches", [])

    if not matches:
        print("No scheduled fixtures returned by the API.")
        return

    # ---------------------------------------------------------
    # Determine the correct season for EVERY match before
    # connecting to the database.
    # ---------------------------------------------------------
    prepared_matches = []
    season_counts = {}

    for match in matches:
        season_info = match.get("season") or {}
        start_date = season_info.get("startDate")

        if not start_date:
            raise ValueError(
                f"Missing season.startDate for match "
                f"{match.get('id')}. Aborting before database writes."
            )

        try:
            start_year = datetime.strptime(
                start_date[:10], "%Y-%m-%d"
            ).year
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid season startDate {start_date!r} "
                f"for match {match.get('id')}."
            )

        season_str = f"{start_year}-{start_year + 1}"

        kickoff_utc = match.get("utcDate")
        if not kickoff_utc:
            raise ValueError(
                f"Missing utcDate for match {match.get('id')}."
            )

        prepared_match = dict(match)
        prepared_match["_db_season"] = season_str
        prepared_matches.append(prepared_match)

        season_counts[season_str] = (
            season_counts.get(season_str, 0) + 1
        )

    print(f"Fixtures grouped by season: {season_counts}")

    conn = get_db_connection()
    cursor = None
    synced_count = 0

    try:
        cursor = conn.cursor()

        # Load existing team mappings.
        cursor.execute("SELECT id, name, aliases FROM teams;")

        team_map = {}

        for tid, name, aliases in cursor.fetchall():
            team_map[name.lower().strip()] = tid

            if aliases:
                alias_list = (
                    json.loads(aliases)
                    if isinstance(aliases, str)
                    else aliases
                )

                for alias in alias_list:
                    team_map[alias.lower().strip()] = tid

        for match in prepared_matches:
            home_raw = match["homeTeam"]["name"]
            away_raw = match["awayTeam"]["name"]

            home_id = get_or_create_team(
                cursor, home_raw, team_map
            )
            away_id = get_or_create_team(
                cursor, away_raw, team_map
            )

            matchweek = match.get("matchday") or 1
            kickoff_utc = match["utcDate"]
            season_str = match["_db_season"]

            cursor.execute(
                """
                INSERT INTO matches (
                    season,
                    matchweek,
                    kickoff_utc,
                    home_team_id,
                    away_team_id,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, 'SCHEDULED')
                ON CONFLICT (
                    season, home_team_id, away_team_id
                )
                DO UPDATE SET
                    matchweek = EXCLUDED.matchweek,
                    kickoff_utc = EXCLUDED.kickoff_utc,
                    status = EXCLUDED.status;
                """,
                (
                    season_str,
                    matchweek,
                    kickoff_utc,
                    home_id,
                    away_id,
                ),
            )

            synced_count += 1

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()

    print(
        f"Successfully synced {synced_count} scheduled fixtures "
        f"across {len(season_counts)} season(s)."
    )


if __name__ == "__main__":
    fetch_and_store_live_fixtures()
