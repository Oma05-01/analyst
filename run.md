after each match week
```
python -m src.sync_finished_results --season-start-year 2026 --apply
python -m src.score_predictions
```

(sports) PS C:\Users\user\PycharmProjects\football-analyst-agent> uvicorn src.api.main:app --reload
(sports) PS C:\Users\user\PycharmProjects\football-analyst-agent\frontend> npm run dev
(sports) PS C:\Users\user\PycharmProjects\football-analyst-agent> streamlit run src/lineup_entry_app.py
