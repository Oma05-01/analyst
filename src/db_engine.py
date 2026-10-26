"""Shared SQLAlchemy engine helper (used by pandas.read_sql in the pipeline scripts)."""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_sqlalchemy_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    if db_url.startswith("postgresql://"):
        # pin the psycopg2 driver (newer SQLAlchemy versions may default to psycopg 3)
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url)