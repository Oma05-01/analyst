
import argparse
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

OLD_SEASON = "2025-2026"
NEW_SEASON = "2026-2027"
EXPECTED_MATCHES = 310
EXPECTED_PREDICTIONS = 328

load_dotenv()


def scalar(conn, sql, params=None):
    return conn.execute(text(sql), params or {}).scalar_one()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the repair. Without this flag, perform a dry run.",
    )
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("ERROR: DATABASE_URL is missing from .env")

    engine = create_engine(database_url)

    try:
        context = engine.begin() if args.apply else engine.connect()

        with context as conn:
            if args.apply:
                conn.execute(text("""
                    SELECT id
                    FROM matches
                    WHERE season IN (:old, :new)
                    FOR UPDATE
                """), {"old": OLD_SEASON, "new": NEW_SEASON}).all()

            old_count = scalar(conn, """
                SELECT COUNT(*) FROM matches WHERE season = :season
            """, {"season": OLD_SEASON})

            new_count = scalar(conn, """
                SELECT COUNT(*) FROM matches WHERE season = :season
            """, {"season": NEW_SEASON})

            matching_pairs = scalar(conn, """
                SELECT COUNT(*)
                FROM matches o
                JOIN matches n
                  ON n.home_team_id = o.home_team_id
                 AND n.away_team_id = o.away_team_id
                 AND n.kickoff_utc = o.kickoff_utc
                WHERE o.season = :old
                  AND n.season = :new
            """, {"old": OLD_SEASON, "new": NEW_SEASON})

            kickoff_mismatches = scalar(conn, """
                SELECT COUNT(*)
                FROM matches o
                JOIN matches n
                  ON n.home_team_id = o.home_team_id
                 AND n.away_team_id = o.away_team_id
                WHERE o.season = :old
                  AND n.season = :new
                  AND o.kickoff_utc IS DISTINCT FROM n.kickoff_utc
            """, {"old": OLD_SEASON, "new": NEW_SEASON})

            old_predictions = scalar(conn, """
                SELECT COUNT(*)
                FROM predictions p
                JOIN matches m ON m.id = p.match_id
                WHERE m.season = :season
            """, {"season": OLD_SEASON})

            new_predictions = scalar(conn, """
                SELECT COUNT(*)
                FROM predictions p
                JOIN matches m ON m.id = p.match_id
                WHERE m.season = :season
            """, {"season": NEW_SEASON})

            new_previews = scalar(conn, """
                SELECT COUNT(*)
                FROM match_previews p
                JOIN matches m ON m.id = p.match_id
                WHERE m.season = :season
            """, {"season": NEW_SEASON})

            print("=== REPAIR PRECHECK ===")
            print(f"Old-season matches:       {old_count}")
            print(f"New-season matches:       {new_count}")
            print(f"Matching fixture pairs:   {matching_pairs}")
            print(f"Kickoff mismatches:       {kickoff_mismatches}")
            print(f"Predictions on old rows:  {old_predictions}")
            print(f"Predictions on new rows:  {new_predictions}")
            print(f"Previews on new rows:     {new_previews}")

            if (
                old_count != EXPECTED_MATCHES
                or new_count != EXPECTED_MATCHES
                or matching_pairs != EXPECTED_MATCHES
                or kickoff_mismatches != 0
                or old_predictions != EXPECTED_PREDICTIONS
                or new_predictions != 0
                or new_previews != 0
            ):
                raise RuntimeError(
                    "Preconditions failed. No repair was committed. "
                    "Review the output before proceeding."
                )

            # Confirm no unanticipated tables reference matches.id.
            foreign_keys = conn.execute(text("""
                SELECT
                    source_ns.nspname AS table_schema,
                    source_table.relname AS table_name,
                    source_column.attname AS column_name,
                    constraint_info.conname AS constraint_name
                FROM pg_constraint AS constraint_info
                JOIN pg_class AS source_table
                  ON source_table.oid = constraint_info.conrelid
                JOIN pg_namespace AS source_ns
                  ON source_ns.oid = source_table.relnamespace
                JOIN pg_class AS target_table
                  ON target_table.oid = constraint_info.confrelid
                JOIN pg_namespace AS target_ns
                  ON target_ns.oid = target_table.relnamespace
                JOIN LATERAL unnest(constraint_info.conkey)
                    WITH ORDINALITY AS source_key(attnum, ord)
                  ON TRUE
                JOIN LATERAL unnest(constraint_info.confkey)
                    WITH ORDINALITY AS target_key(attnum, ord)
                  ON target_key.ord = source_key.ord
                JOIN pg_attribute AS source_column
                  ON source_column.attrelid = source_table.oid
                 AND source_column.attnum = source_key.attnum
                JOIN pg_attribute AS target_column
                  ON target_column.attrelid = target_table.oid
                 AND target_column.attnum = target_key.attnum
                WHERE constraint_info.contype = 'f'
                  AND target_ns.nspname = 'public'
                  AND target_table.relname = 'matches'
                  AND target_column.attname = 'id'
                ORDER BY
                    source_ns.nspname,
                    source_table.relname,
                    source_column.attname;
            """)).mappings().all()

            actual_references = {
                (row["table_name"], row["column_name"])
                for row in foreign_keys
            }

            print("\nActual foreign keys referencing matches.id:")
            for row in foreign_keys:
                print(
                    f"  {row['table_schema']}."
                    f"{row['table_name']}."
                    f"{row['column_name']} "
                    f"({row['constraint_name']})"
                )

            allowed = {
                ("predictions", "match_id"),
                ("match_previews", "match_id"),
            }

            unexpected = actual_references - allowed

            if unexpected:
                raise RuntimeError(
                    "Unexpected tables reference matches.id: "
                    f"{unexpected}"
                )

            print("\nPreconditions passed.")

            if not args.apply:
                print("DRY RUN ONLY: nothing was changed.")
                print(
                    "Review the results, then rerun with --apply "
                    "to perform the repair."
                )
                return

            # Delete only the duplicate rows, which have no predictions
            # or previews. Keep the original match IDs intact.
            deleted = conn.execute(text("""
                DELETE FROM matches n
                USING matches o
                WHERE o.season = :old
                  AND n.season = :new
                  AND n.home_team_id = o.home_team_id
                  AND n.away_team_id = o.away_team_id
                  AND n.kickoff_utc = o.kickoff_utc
                  AND NOT EXISTS (
                      SELECT 1 FROM predictions p
                      WHERE p.match_id = n.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM match_previews p
                      WHERE p.match_id = n.id
                  )
                RETURNING n.id
            """), {"old": OLD_SEASON, "new": NEW_SEASON}).all()

            if len(deleted) != EXPECTED_MATCHES:
                raise RuntimeError(
                    f"Expected to delete {EXPECTED_MATCHES} duplicates; "
                    f"deleted {len(deleted)}. Transaction will roll back."
                )

            updated = conn.execute(text("""
                UPDATE matches
                SET season = :new
                WHERE season = :old
            """), {"old": OLD_SEASON, "new": NEW_SEASON})

            if updated.rowcount != EXPECTED_MATCHES:
                raise RuntimeError(
                    f"Expected to relabel {EXPECTED_MATCHES} originals; "
                    f"updated {updated.rowcount}. Transaction will roll back."
                )

            print("\nRepair completed within the transaction.")
            print(f"Duplicate rows removed: {len(deleted)}")
            print(f"Original rows relabelled: {updated.rowcount}")
            print(f"Original prediction rows preserved: {old_predictions}")

        if args.apply:
            print("Transaction committed successfully.")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
