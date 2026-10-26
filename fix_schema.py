import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
# Ensure your .env has your DATABASE_URL set
engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE predictions ALTER COLUMN p_away TYPE DOUBLE PRECISION;"))
    conn.execute(text("ALTER TABLE predictions ALTER COLUMN expected_home_goals TYPE DOUBLE PRECISION;"))
    conn.execute(text("ALTER TABLE predictions ALTER COLUMN expected_away_goals TYPE DOUBLE PRECISION;"))
    conn.commit()

print("Schema updated successfully.")