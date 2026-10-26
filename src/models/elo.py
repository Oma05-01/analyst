import math
import pandas as pd
import numpy as np

class EloModel:
    def __init__(self, k_factor=20.0, home_advantage=65.0, draw_width=100.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.draw_width = draw_width
        self.ratings = {}  # team_id -> current_elo

    def get_rating(self, team_id):
        return self.ratings.get(team_id, 1500.0)

    def predict_probs(self, home_team_id, away_team_id):
        r_home = self.get_rating(home_team_id) + self.home_advantage
        r_away = self.get_rating(away_team_id)

        dr = r_home - r_away

        # Win probability using logistic function
        win_prob = 1.0 / (1.0 + math.pow(10, -dr / 400.0))
        
        # Empirical draw margin estimation
        p_draw = max(0.10, 0.28 - (abs(dr) / 1500.0))
        p_home = win_prob * (1.0 - p_draw)
        p_away = (1.0 - win_prob) * (1.0 - p_draw)

        # Normalize probabilities to sum strictly to 1.0
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def update_ratings(self, home_team_id, away_team_id, actual_outcome):
        r_home_effective = self.get_rating(home_team_id) + self.home_advantage
        r_away = self.get_rating(away_team_id)

        e_home = 1.0 / (1.0 + math.pow(10, (r_away - r_home_effective) / 400.0))
        
        if actual_outcome == 'HOME':
            s_home = 1.0
        elif actual_outcome == 'DRAW':
            s_home = 0.5
        else:
            s_home = 0.0

        r_home_old = self.get_rating(home_team_id)
        r_away_old = self.get_rating(away_team_id)

        self.ratings[home_team_id] = r_home_old + self.k_factor * (s_home - e_home)
        self.ratings[away_team_id] = r_away_old + self.k_factor * ((1.0 - s_home) - (1.0 - e_home))