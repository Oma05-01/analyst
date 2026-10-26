
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise SystemExit("DATABASE_URL is missing from .env")

engine = create_engine(database_url)

query = text("""
    SELECT
        season,
        COUNT(*) AS total_matches,
        COUNT(*) FILTER (
            WHERE home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND status = 'FINISHED'
        ) AS finished_with_scores,
        COUNT(*) FILTER (
            WHERE home_goals IS NULL
               OR away_goals IS NULL
        ) AS missing_scores,
        COUNT(DISTINCT home_team_id) AS home_teams,
        COUNT(DISTINCT away_team_id) AS away_teams,
        MIN(kickoff_utc) AS first_kickoff,
        MAX(kickoff_utc) AS last_kickoff
    FROM matches
    GROUP BY season
    ORDER BY season;
""")

try:
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

        print("\n=== HISTORICAL MATCH COVERAGE ===\n")

        if not rows:
            print("No match records found.")
        else:
            for row in rows:
                print(
                    f"Season: {row['season']}\n"
                    f"  Total matches:       {row['total_matches']}\n"
                    f"  Finished with scores:{row['finished_with_scores']:>8}\n"
                    f"  Missing scores:      {row['missing_scores']}\n"
                    f"  Distinct home teams: {row['home_teams']}\n"
                    f"  Distinct away teams: {row['away_teams']}\n"
                    f"  First kickoff:       {row['first_kickoff']}\n"
                    f"  Last kickoff:        {row['last_kickoff']}\n"
                )
finally:
    engine.dispose()
