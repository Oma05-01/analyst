import pandas as pd

df = pd.read_csv("data/raw/E0_2425.csv", encoding="latin1")

print(
    df.groupby(
        ["competition_code", "competition_name", "season"],
        dropna=False
    )
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nCompetition codes:")
print(df["competition_code"].value_counts(dropna=False).to_string())