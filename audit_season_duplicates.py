
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise SystemExit("ERROR: DATABASE_URL is missing from .env")

engine = create_engine(database_url)

season_summary = text("""
    SELECT
        m.season,
        COUNT(*) AS total_matches,
        COUNT(DISTINCT (m.home_team_id, m.away_team_id))
            AS unique_fixtures,
        COUNT(p.match_id) AS prediction_rows
    FROM matches m
    LEFT JOIN predictions p ON p.match_id = m.id
    WHERE m.season IN ('2025-2026', '2026-2027')
    GROUP BY m.season
    ORDER BY m.season;
""")

duplicates = text("""
    SELECT
        old.id AS old_match_id,
        new.id AS new_match_id,
        old.season AS old_season,
        new.season AS new_season,
        ht.name AS home_team,
        at.name AS away_team,
        old.kickoff_utc AS old_kickoff,
        new.kickoff_utc AS new_kickoff,
        old.status AS old_status,
        new.status AS new_status,
        (
            SELECT COUNT(*)
            FROM predictions p
            WHERE p.match_id = old.id
        ) AS old_prediction_count,
        (
            SELECT COUNT(*)
            FROM predictions p
            WHERE p.match_id = new.id
        ) AS new_prediction_count
    FROM matches old
    JOIN matches new
      ON new.home_team_id = old.home_team_id
     AND new.away_team_id = old.away_team_id
     AND new.season = '2026-2027'
    JOIN teams ht ON ht.id = old.home_team_id
    JOIN teams at ON at.id = old.away_team_id
    WHERE old.season = '2025-2026'
    ORDER BY new.kickoff_utc;
""")

try:
    with engine.connect() as conn:
        print("\n=== SEASON SUMMARY ===")
        rows = conn.execute(season_summary).mappings().all()

        if not rows:
            print("No matching season records found.")

        for row in rows:
            print(dict(row))

        print("\n=== MATCHES PRESENT IN BOTH SEASONS ===")
        rows = conn.execute(duplicates).mappings().all()

        print(f"Matching home/away pairs: {len(rows)}")

        for row in rows:
            print(dict(row))

finally:
    engine.dispose()
