
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise SystemExit("ERROR: DATABASE_URL is missing from .env")

engine = create_engine(database_url)

queries = {
    "SEASON TOTALS": """
        SELECT
            m.season,
            COUNT(*) AS total_matches,
            COUNT(DISTINCT (
                m.home_team_id, m.away_team_id
            )) AS unique_fixtures,
            COUNT(*) FILTER (
                WHERE COALESCE(p.prediction_count, 0) > 0
            ) AS matches_with_predictions,
            SUM(COALESCE(p.prediction_count, 0))
                AS total_prediction_rows
        FROM matches m
        LEFT JOIN (
            SELECT match_id, COUNT(*) AS prediction_count
            FROM predictions
            GROUP BY match_id
        ) p ON p.match_id = m.id
        WHERE m.season IN ('2025-2026', '2026-2027')
        GROUP BY m.season
        ORDER BY m.season;
    """,

    "DUPLICATE SUMMARY": """
        WITH prediction_counts AS (
            SELECT match_id, COUNT(*) AS n
            FROM predictions
            GROUP BY match_id
        ),
        old_rows AS (
            SELECT
                m.id,
                m.home_team_id,
                m.away_team_id,
                m.kickoff_utc,
                COALESCE(p.n, 0) AS prediction_count
            FROM matches m
            LEFT JOIN prediction_counts p ON p.match_id = m.id
            WHERE m.season = '2025-2026'
        ),
        new_rows AS (
            SELECT
                m.id,
                m.home_team_id,
                m.away_team_id,
                m.kickoff_utc,
                COALESCE(p.n, 0) AS prediction_count
            FROM matches m
            LEFT JOIN prediction_counts p ON p.match_id = m.id
            WHERE m.season = '2026-2027'
        )
        SELECT
            COUNT(*) AS duplicate_pairs,
            COUNT(*) FILTER (
                WHERE o.prediction_count > 0
            ) AS old_matches_with_predictions,
            COALESCE(SUM(o.prediction_count), 0)
                AS predictions_on_old_matches,
            COUNT(*) FILTER (
                WHERE n.prediction_count > 0
            ) AS new_matches_with_predictions,
            COUNT(*) FILTER (
                WHERE o.kickoff_utc IS DISTINCT FROM n.kickoff_utc
            ) AS kickoff_mismatches
        FROM old_rows o
        JOIN new_rows n
          ON n.home_team_id = o.home_team_id
         AND n.away_team_id = o.away_team_id;
    """,

    "FOREIGN KEY REFERENCES TO MATCHES": """
        SELECT
            tc.table_schema,
            tc.table_name,
            kcu.column_name,
            tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.constraint_schema = kcu.constraint_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.constraint_schema = tc.constraint_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND ccu.table_schema = 'public'
          AND ccu.table_name = 'matches'
          AND ccu.column_name = 'id'
        ORDER BY tc.table_name, kcu.column_name;
    """,
}

try:
    with engine.connect() as conn:
        for title, sql in queries.items():
            print(f"\n=== {title} ===")
            rows = conn.execute(text(sql)).mappings().all()

            if not rows:
                print("No rows returned.")
            else:
                for row in rows:
                    print(dict(row))
finally:
    engine.dispose()
