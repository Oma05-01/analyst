import pandas as pd

class HistoricalFrequencyModel:
    """
    Baseline predictor returning historical win/draw/loss proportions 
    computed strictly from matches prior to the target match date.
    """
    def __init__(self):
        # Default global English Premier League historical proportions fallback
        self.p_home = 0.45
        self.p_draw = 0.25
        self.p_away = 0.30

    def fit(self, historical_matches_df):
        if historical_matches_df.empty:
            return

        total = len(historical_matches_df)
        home_wins = len(historical_matches_df[historical_matches_df['home_goals'] > historical_matches_df['away_goals']])
        draws = len(historical_matches_df[historical_matches_df['home_goals'] == historical_matches_df['away_goals']])
        away_wins = total - home_wins - draws

        self.p_home = home_wins / total
        self.p_draw = draws / total
        self.p_away = away_wins / total

    def predict(self):
        return self.p_home, self.p_draw, self.p_away