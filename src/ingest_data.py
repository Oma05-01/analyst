import io
import json
import os
import re
import pandas as pd
from db import get_db_connection

# Target 2024-2025
SEASONS = [("2425", "2024-2025")]

def load_team_mappings(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, name, aliases FROM teams;")
        rows = cursor.fetchall()
    finally:
        cursor.close()

    mapping = {}

    def register_name(name, team_id):
        if not name:
            return

        name = str(name).strip()
        keys = {name.casefold(), clean_team_name(name)}

        for key in keys:
            if not key:
                continue

            existing_id = mapping.get(key)

            if existing_id is not None and existing_id != team_id:
                raise ValueError(
                    f"Ambiguous team mapping for {key!r}: "
                    f"team IDs {existing_id} and {team_id}"
                )

            mapping[key] = team_id

    for team_id, name, aliases in rows:
        register_name(name, team_id)

        if aliases:
            alias_list = (
                json.loads(aliases)
                if isinstance(aliases, str)
                else aliases
            )

            for alias in alias_list:
                register_name(alias, team_id)

    return mapping

def clean_team_name(name):
    """Normalize team names by stripping trailing FC/AFC and extra spaces."""
    name = str(name).strip()
    name = re.sub(r'\s+FC$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^AFC\s+', '', name, flags=re.IGNORECASE)
    return name.strip().lower()

def parse_date(date_str):
    date_str = str(date_str).strip()
    try:
        return pd.to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Unknown date format '{date_str}': {e}")

def process_and_store_season(conn, team_map, season_str, csv_bytes):
    # ---------------------------------------------------------
    # 1. Read and validate the source structure
    # ---------------------------------------------------------
    df = pd.read_csv(
        io.BytesIO(csv_bytes),
        encoding="utf-8-sig",
    )
    df.columns = [str(col).strip().lower() for col in df.columns]

    required_columns = {
        "competition_code",
        "season",
        "match_id",
        "matchday",
        "date_utc",
        "home_team",
        "away_team",
        "fulltime_home",
        "fulltime_away",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # 2. Filter to the intended competition and season
    # ---------------------------------------------------------
    source_season = season_str.replace("-", "/")

    df = df[
        df["competition_code"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("PL")
        &
        df["season"]
        .astype(str)
        .str.strip()
        .eq(source_season)
    ].copy()

    if len(df) != 380:
        raise ValueError(
            f"Expected 380 Premier League matches for {source_season}; "
            f"found {len(df)}. No data was written."
        )

    if df["match_id"].isna().any():
        raise ValueError("Some source matches have no match_id.")

    if df["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id values found.")

    clubs = set(df["home_team"].dropna().astype(str))
    clubs.update(df["away_team"].dropna().astype(str))

    if len(clubs) != 20:
        raise ValueError(
            f"Expected 20 distinct Premier League clubs; found {len(clubs)}."
        )

    if df[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Some matches have a missing home or away team.")

    # ---------------------------------------------------------
    # 3. Preserve the real kickoff timestamps
    # ---------------------------------------------------------
    df["_kickoff_utc"] = pd.to_datetime(
        df["date_utc"],
        utc=True,
        errors="raise",
    )

    if df["_kickoff_utc"].isna().any():
        raise ValueError("Some matches have missing kickoff timestamps.")

    # Use the actual matchday, not an estimate based on row position.
    df["_matchweek"] = pd.to_numeric(
        df["matchday"],
        errors="coerce",
    )

    if (
        df["_matchweek"].isna().any()
        or (df["_matchweek"] % 1 != 0).any()
    ):
        raise ValueError("Some matches have invalid matchday values.")

    # ---------------------------------------------------------
    # 4. Validate full-time scores
    # ---------------------------------------------------------
    for column in ("fulltime_home", "fulltime_away"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

        invalid = (
            df[column].isna()
            | (df[column] < 0)
            | (df[column] % 1 != 0)
        )

        if invalid.any():
            raise ValueError(
                f"Invalid or missing scores in {column}. "
                "This historical-season import expects completed matches."
            )

    # ---------------------------------------------------------
    # 5. Resolve every club before touching the database
    # ---------------------------------------------------------
    def resolve_team_id(raw_name):
        raw_name = str(raw_name).strip()

        for key in (
            raw_name.casefold(),
            clean_team_name(raw_name),
        ):
            team_id = team_map.get(key)

            if team_id is not None:
                return team_id

        return None

    club_ids = {}
    unmapped_teams = set()

    for club in clubs:
        team_id = resolve_team_id(club)

        if team_id is None:
            unmapped_teams.add(club)
        else:
            club_ids[club] = team_id

    if unmapped_teams:
        raise ValueError(
            "Premier League clubs missing from the database team mapping: "
            f"{sorted(unmapped_teams)}. No matches were written."
        )

    df["_home_team_id"] = df["home_team"].map(club_ids)
    df["_away_team_id"] = df["away_team"].map(club_ids)

    if (df["_home_team_id"] == df["_away_team_id"]).any():
        raise ValueError("A match has the same home and away team.")

    if df.duplicated(["_home_team_id", "_away_team_id"]).any():
        raise ValueError("Duplicate home/away fixture pairs found.")

    # ---------------------------------------------------------
    # 6. Optional statistics: missing means unknown, not zero
    # ---------------------------------------------------------
    def optional_int(row, *column_names):
        for name in column_names:
            if name in df.columns and pd.notna(row[name]):
                value = pd.to_numeric(row[name], errors="raise")

                if value % 1 != 0:
                    raise ValueError(
                        f"Non-integer statistic {name}={value!r}"
                    )

                return int(value)

        return None

    def total_cards(row, yellow_columns, red_columns):
        yellow = optional_int(row, *yellow_columns)
        red = optional_int(row, *red_columns)

        if yellow is None or red is None:
            return None

        return yellow + red

    # ---------------------------------------------------------
    # 7. Upsert only after all pre-write validation succeeds
    # ---------------------------------------------------------
    cursor = conn.cursor()
    processed_count = 0

    try:
        for _, row in df.iterrows():
            home_id = int(row["_home_team_id"])
            away_id = int(row["_away_team_id"])

            home_goals = int(row["fulltime_home"])
            away_goals = int(row["fulltime_away"])

            kickoff_utc = row["_kickoff_utc"].to_pydatetime()
            matchweek = int(row["_matchweek"])

            home_shots = optional_int(
                row, "hs", "home_shots"
            )
            away_shots = optional_int(
                row, "as", "away_shots"
            )
            home_corners = optional_int(
                row, "hc", "home_corners"
            )
            away_corners = optional_int(
                row, "ac", "away_corners"
            )

            home_cards = total_cards(
                row,
                ("hy", "home_yellow_cards"),
                ("hr", "home_red_cards"),
            )
            away_cards = total_cards(
                row,
                ("ay", "away_yellow_cards"),
                ("ar", "away_red_cards"),
            )

            referee = (
                str(row["referee"]).strip()
                if "referee" in df.columns
                and pd.notna(row["referee"])
                else None
            )

            cursor.execute(
                """
                INSERT INTO matches (
                    season,
                    matchweek,
                    kickoff_utc,
                    home_team_id,
                    away_team_id,
                    status,
                    home_goals,
                    away_goals,
                    home_shots,
                    away_shots,
                    home_corners,
                    away_corners,
                    home_cards,
                    away_cards,
                    referee
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (
                    season, home_team_id, away_team_id
                )
                DO UPDATE SET
                    matchweek = EXCLUDED.matchweek,
                    kickoff_utc = EXCLUDED.kickoff_utc,
                    status = EXCLUDED.status,
                    home_goals = EXCLUDED.home_goals,
                    away_goals = EXCLUDED.away_goals,
                    home_shots = COALESCE(
                        EXCLUDED.home_shots, matches.home_shots
                    ),
                    away_shots = COALESCE(
                        EXCLUDED.away_shots, matches.away_shots
                    ),
                    home_corners = COALESCE(
                        EXCLUDED.home_corners, matches.home_corners
                    ),
                    away_corners = COALESCE(
                        EXCLUDED.away_corners, matches.away_corners
                    ),
                    home_cards = COALESCE(
                        EXCLUDED.home_cards, matches.home_cards
                    ),
                    away_cards = COALESCE(
                        EXCLUDED.away_cards, matches.away_cards
                    ),
                    referee = COALESCE(
                        EXCLUDED.referee, matches.referee
                    );
                """,
                (
                    season_str,
                    matchweek,
                    kickoff_utc,
                    home_id,
                    away_id,
                    "FINISHED",
                    home_goals,
                    away_goals,
                    home_shots,
                    away_shots,
                    home_corners,
                    away_corners,
                    home_cards,
                    away_cards,
                    referee,
                ),
            )

            processed_count += 1

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()

    print(
        f"Successfully upserted {processed_count} "
        f"Premier League matches for {season_str}."
    )

    return processed_count

def download_and_ingest():
    raw_dir = os.path.join("data", "raw")
    raw_filepath = os.path.join(raw_dir, "E0_2425.csv")

    if not os.path.exists(raw_filepath):
        print(f"Error: {raw_filepath} not found.")
        return

    with open(raw_filepath, "rb") as f:
        csv_bytes = f.read()

    print("Connecting to database...")
    conn = get_db_connection()
    team_map = load_team_mappings(conn)

    print("Ingesting 2024-2025...")
    count = process_and_store_season(conn, team_map, "2024-2025", csv_bytes)
    conn.close()

    print(f"\nDone! Processed {count} matches from 2024-2025.")

if __name__ == "__main__":
    download_and_ingest()