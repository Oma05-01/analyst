import pandas as pd
from db import get_db_connection
from metrics import calculate_brier_score, calculate_rps, calculate_log_loss
from models.baseline import HistoricalFrequencyModel
from models.elo import EloModel
from models.dixon_coles import DixonColesModel

def load_all_finished_matches():
    conn = get_db_connection()
    query = """
        SELECT id, season, matchweek, kickoff_utc, home_team_id, away_team_id, home_goals, away_goals
        FROM matches
        WHERE status = 'FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY kickoff_utc ASC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def run_evaluation():
    df_matches = load_all_finished_matches()

    def get_outcome(row):
        if row['home_goals'] > row['away_goals']:
            return 'HOME'
        elif row['home_goals'] < row['away_goals']:
            return 'AWAY'
        return 'DRAW'

    df_matches['outcome'] = df_matches.apply(get_outcome, axis=1)

    seasons = df_matches['season'].unique()
    test_seasons = [s for s in seasons if s != seasons[0]]

    elo = EloModel(k_factor=20.0, home_advantage=65.0)
    
    # Pre-train Elo on season 1
    burn_in_matches = df_matches[df_matches['season'] == seasons[0]]
    for _, match in burn_in_matches.iterrows():
        elo.update_ratings(match['home_team_id'], match['away_team_id'], match['outcome'])

    results_baseline, results_elo, results_dc = [], [], []
    baseline_model = HistoricalFrequencyModel()
    dc_model = DixonColesModel()

    for season in test_seasons:
        print(f"Evaluating season {season}...")
        season_matches = df_matches[df_matches['season'] == season]
        matchweeks = sorted(season_matches['matchweek'].unique())

        for mw in matchweeks:
            mw_matches = season_matches[season_matches['matchweek'] == mw]
            if mw_matches.empty:
                continue

            first_kickoff = mw_matches['kickoff_utc'].min()

            # Train models strictly on past matches
            train_df = df_matches[df_matches['kickoff_utc'] < first_kickoff]
            baseline_model.fit(train_df)
            dc_model.fit(train_df)

            for _, match in mw_matches.iterrows():
                outcome = match['outcome']
                h_id, a_id = match['home_team_id'], match['away_team_id']

                # 0. Baseline
                b_ph, b_pd, b_pa = baseline_model.predict()
                results_baseline.append({
                    'brier': calculate_brier_score(b_ph, b_pd, b_pa, outcome),
                    'rps': calculate_rps(b_ph, b_pd, b_pa, outcome),
                    'log_loss': calculate_log_loss(b_ph, b_pd, b_pa, outcome)
                })

                # 1. Elo
                e_ph, e_pd, e_pa = elo.predict_probs(h_id, a_id)
                results_elo.append({
                    'brier': calculate_brier_score(e_ph, e_pd, e_pa, outcome),
                    'rps': calculate_rps(e_ph, e_pd, e_pa, outcome),
                    'log_loss': calculate_log_loss(e_ph, e_pd, e_pa, outcome)
                })
                elo.update_ratings(h_id, a_id, outcome)

                # 2. Dixon-Coles
                dc_ph, dc_pd, dc_pa, _, _, _ = dc_model.predict_probs(h_id, a_id)
                results_dc.append({
                    'brier': calculate_brier_score(dc_ph, dc_pd, dc_pa, outcome),
                    'rps': calculate_rps(dc_ph, dc_pd, dc_pa, outcome),
                    'log_loss': calculate_log_loss(dc_ph, dc_pd, dc_pa, outcome)
                })

    df_b = pd.DataFrame(results_baseline)
    df_e = pd.DataFrame(results_elo)
    df_dc = pd.DataFrame(results_dc)

    print("\n================ WALK-FORWARD MODEL COMPARISON ================")
    print(f"Evaluated Matches: {len(df_b)}\n")
    print(f"MODEL                   | BRIER SCORE | RPS SCORE | LOG LOSS")
    print(f"-----------------------------------------------------------")
    print(f"0. Historical Baseline  | {df_b['brier'].mean():.4f}      | {df_b['rps'].mean():.4f}    | {df_b['log_loss'].mean():.4f}")
    print(f"1. Elo Rating Model     | {df_e['brier'].mean():.4f}      | {df_e['rps'].mean():.4f}    | {df_e['log_loss'].mean():.4f}")
    print(f"2. Dixon-Coles Model    | {df_dc['brier'].mean():.4f}     | {df_dc['rps'].mean():.4f}    | {df_dc['log_loss'].mean():.4f}")
    print("==============================================================")

if __name__ == "__main__":
    run_evaluation()