import json
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Execute schema SQL
    with open("schema.sql", "r") as f:
        schema_sql = f.read()
    cursor.execute(schema_sql)
    print("Database schema created successfully.")

    # Seed teams mapping
    with open("data/team_mapping.json", "r") as f:
        teams_data = json.load(f)

    for canonical_name, data in teams_data.items():
        cursor.execute(
            """
            INSERT INTO teams (name, short_code, primary_colour, secondary_colour, aliases)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                short_code = EXCLUDED.short_code,
                primary_colour = EXCLUDED.primary_colour,
                secondary_colour = EXCLUDED.secondary_colour,
                aliases = EXCLUDED.aliases;
            """,
            (
                canonical_name,
                data["short_code"],
                data["primary_colour"],
                data["secondary_colour"],
                json.dumps(data["aliases"])
            )
        )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Successfully seeded {len(teams_data)} teams into database.")

if __name__ == "__main__":
    init_db()