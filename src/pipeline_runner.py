from fetch_upcoming import fetch_and_store_live_fixtures
from generate_predictions import generate_and_lock_predictions
from score_predictions import score_unscored_predictions

def run_full_pipeline():
    print("\n================ STARTING PIPELINE RUN ================")
    
    print("\n[1/3] Fetching latest fixtures & unplayed matches...")
    fetch_and_store_live_fixtures()

    print("\n[2/3] Generating and locking predictions for upcoming matches...")
    generate_and_lock_predictions()

    print("\n[3/3] Scoring completed match predictions...")
    score_unscored_predictions()

    print("\n================ PIPELINE RUN COMPLETE ================\n")

if __name__ == "__main__":
    run_full_pipeline()