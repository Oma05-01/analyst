from src.fetch_upcoming import fetch_and_store_live_fixtures
from src.generate_predictions import main as generate_predictions_main
from src.score_predictions import score_unscored_predictions
from src.db_engine import get_sqlalchemy_engine

def run_full_pipeline():
    print("\n================ STARTING PIPELINE RUN ================")

    print("\n[1/3] Fetching latest fixtures & unplayed matches...")
    fetch_and_store_live_fixtures()

    print("\n[2/3] Generating and locking predictions...")
    result = generate_predictions_main()

    if result != 0:
        raise RuntimeError(
            f"Prediction generation failed with exit code {result}"
        )

    print("\n[3/3] Scoring completed match predictions...")

    engine = get_sqlalchemy_engine()
    try:
        score_unscored_predictions(engine)
    finally:
        engine.dispose()

    print("\n================ PIPELINE RUN COMPLETE ================\n")


if __name__ == "__main__":
    run_full_pipeline()