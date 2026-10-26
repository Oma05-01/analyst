from src.db import get_db_connection

def apply_migration():
    conn = get_db_connection()
    cursor = conn.cursor()

    print("Adding payload_hash and match_preview_text columns...")
    cursor.execute("""
        ALTER TABLE predictions 
        ADD COLUMN IF NOT EXISTS payload_hash TEXT,
        ADD COLUMN IF NOT EXISTS match_preview_text TEXT;
    """)

    print("Creating PostgreSQL trigger for immutability locking...")
    cursor.execute("""
        CREATE OR REPLACE FUNCTION prevent_locked_prediction_update()
        RETURNS TRIGGER AS $$
        BEGIN
            IF OLD.locked_at IS NOT NULL AND OLD.locked_at <= NOW() THEN
                RAISE EXCEPTION 'Prediction ID % is locked and cannot be updated or modified.', OLD.id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS trg_immutable_predictions ON predictions;

        CREATE TRIGGER trg_immutable_predictions
        BEFORE UPDATE OR DELETE ON predictions
        FOR EACH ROW
        EXECUTE FUNCTION prevent_locked_prediction_update();
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database migration and immutability trigger applied successfully!")

if __name__ == "__main__":
    apply_migration()