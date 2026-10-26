from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

query = text("""
    SELECT
        column_name,
        data_type,
        numeric_precision,
        numeric_scale
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'predictions'
      AND column_name IN (
          'p_home',
          'p_draw',
          'p_away',
          'expected_home_goals',
          'expected_away_goals'
      )
    ORDER BY column_name;
""")

with engine.connect() as conn:
    for row in conn.execute(query):
        print(row)