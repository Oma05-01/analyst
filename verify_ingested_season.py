
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def main():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        sys.exit("ERROR: DATABASE_URL is missing from your .env file.")

    engine = create_engine(database_url)

    query = text("""
        WITH season_matches AS (
            SELECT *
            FROM matches
            WHERE season = :season
        ),
        all_teams AS (
            SELECT home_team_id AS team_id
            FROM season_matches

            UNION

            SELECT away_team_id AS team_id
            FROM season_matches
        )
        SELECT
            :season AS season,

            COUNT(*) AS total_matches,

            COUNT(DISTINCT (
                home_team_id, away_team_id
            )) AS unique_fixtures,

            (SELECT COUNT(*) FROM all_teams)
                AS distinct_teams,

            COUNT(DISTINCT home_team_id)
                AS distinct_home_teams,

            COUNT(DISTINCT away_team_id)
                AS distinct_away_teams,

            COUNT(*) FILTER (
                WHERE status = 'FINISHED'
            ) AS finished_matches,

            COUNT(*) FILTER (
                WHERE home_goals IS NULL
                   OR away_goals IS NULL
            ) AS matches_missing_scores,

            MIN(kickoff_utc) AS first_kickoff,
            MAX(kickoff_utc) AS last_kickoff

        FROM season_matches
        GROUP BY season
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(
                query,
                {"season": "2024-2025"},
            ).mappings().one_or_none()

            if result is None:
                print("No matches found for season 2024-2025.")
                return

            print("\n=== DATABASE VERIFICATION ===")
            for key, value in result.items():
                print(f"{key}: {value}")

            print("\n=== BASIC VALIDATION ===")
            checks = {
                "380 matches": result["total_matches"] == 380,
                "380 unique fixtures": result["unique_fixtures"] == 380,
                "20 distinct clubs": result["distinct_teams"] == 20,
                "20 home teams": result["distinct_home_teams"] == 20,
                "20 away teams": result["distinct_away_teams"] == 20,
                "380 finished matches": result["finished_matches"] == 380,
                "No missing scores": result["matches_missing_scores"] == 0,
            }

            for description, passed in checks.items():
                print(f"{'PASS' if passed else 'CHECK'}: {description}")

    except SQLAlchemyError as exc:
        sys.exit(f"Database query failed: {exc}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
