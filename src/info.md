1. Your existing baseline

src/models/dixon_coles.py

This contains the Dixon–Coles model's fit() and predict_probs() methods. This is the model we want to preserve as our benchmark.




2. Your existing lineup-aware model

src/evaluate_player_availability.py

This contains a training method and a prediction method that accepts home and away lineup features. We can study this as a pattern for incorporating additional information.




3. Historical training data

src/generate_lineup_forecast.py — load_finished_training()

This loads completed matches before the forecast cutoff. We need a similar historical-data foundation for calculating recent form and head-to-head statistics without accidentally using future results.




4. Where forecasts are assembled

src/generate_lineup_forecast.py — make_forecast()

This connects the fixture, historical data, lineup snapshot and model predictions. It's a likely integration point for our experimental features.