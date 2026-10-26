"""Constants shared by the pipeline scripts and the API."""

MODEL_VERSION = "dixon_coles_v2"          # the primary, publicly shown model
BASELINE_VERSION = "baseline_poisson_v1"  # league-average baseline logged beside it

# Check these against what your ingest writes into matches.status
UPCOMING_STATUSES = ["SCHEDULED", "TIMED"]
SETTLED_STATUSES = ["FINISHED", "POSTPONED", "CANCELLED", "AWARDED", "SUSPENDED"]