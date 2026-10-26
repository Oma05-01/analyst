
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import make_url


def main():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        sys.exit("ERROR: DATABASE_URL is missing from .env")

    pg_dump = shutil.which("pg_dump")
    if not pg_dump:
        sys.exit(
            "ERROR: pg_dump was not found. "
            "Check your PostgreSQL installation and PATH."
        )

    url = make_url(database_url)

    if url.get_backend_name() != "postgresql":
        sys.exit("ERROR: DATABASE_URL must point to PostgreSQL.")

    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"football_analyst_{timestamp}.dump"

    command = [
        pg_dump,
        "--format=custom",
        "--file", str(backup_path),
        "--dbname", url.database,
    ]

    if url.host:
        command.extend(["--host", url.host])
    if url.port:
        command.extend(["--port", str(url.port)])
    if url.username:
        command.extend(["--username", url.username])

    env = os.environ.copy()

    # Pass credentials privately to the child process.
    if url.password is not None:
        env["PGPASSWORD"] = url.password

    query_params = dict(url.query)
    if query_params.get("sslmode"):
        env["PGSSLMODE"] = str(query_params["sslmode"])

    try:
        subprocess.run(command, check=True, env=env)

        if not backup_path.exists() or backup_path.stat().st_size == 0:
            sys.exit("ERROR: Backup file is missing or empty.")

        print("Database backup completed successfully.")
        print(f"File: {backup_path}")
        print(f"Size: {backup_path.stat().st_size:,} bytes")

    except subprocess.CalledProcessError as exc:
        sys.exit(f"ERROR: pg_dump failed with exit code {exc.returncode}")


if __name__ == "__main__":
    main()
