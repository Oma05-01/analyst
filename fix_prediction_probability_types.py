
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise SystemExit("ERROR: DATABASE_URL is missing from .env")

engine = create_engine(database_url)

try:
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE public.predictions
            ALTER COLUMN p_home TYPE DOUBLE PRECISION
            USING p_home::DOUBLE PRECISION
        """))

        conn.execute(text("""
            ALTER TABLE public.predictions
            ALTER COLUMN p_draw TYPE DOUBLE PRECISION
            USING p_draw::DOUBLE PRECISION
        """))

    print("Probability column types updated successfully.")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'predictions'
              AND column_name IN (
                  'p_home', 'p_draw', 'p_away'
              )
            ORDER BY column_name
        """)).all()

        for row in rows:
            print(row)

finally:
    engine.dispose()
