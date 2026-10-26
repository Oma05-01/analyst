"""
Generate and lock pre-match predictions.

Run from the repository root:   python -m src.generate_predictions
Run the migration first:        python -m src.migrations.upgrade_schema

Design rules
  * Locking comes first and never calls the LLM. Previews are generated afterwards,
    stored in match_previews, and a failed preview never blocks or undoes a lock.
  * Each match is committed on its own, so one bad match cannot discard the rest.
  * locked_at is assigned by the database (see the migration); the job never sends it.
  * Only matches with no prediction yet for this model_version are processed, so
    repeated runs do no training work and make no LLM calls for locked matches.
  * Unknown teams are skipped and logged. Numbers are never invented.
  * A league-average baseline is logged beside every prediction for comparison.
  * The hash covers the prediction's content and inputs only (not locked_at).
"""
import hashlib
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from src.db import get_db_connection
from src.db_engine import get_sqlalchemy_engine
from src.models.dixon_coles import (DixonColesModel, UnknownTeamError, outcome_probs,
                                    score_matrix)

MODEL_VERSION = "dixon_coles_v2"
BASELINE_VERSION = "baseline_poisson_v1"
TIME_DECAY = 0.001
RIDGE = 1.0
MAX_GOALS = 9
LOCK_WINDOW_HOURS = 72          # lock each match the first run that sees it within this window
MIN_TRAIN_MATCHES = 200
PREVIEW_BATCH_LIMIT = 20

# Check these against what your ingest actually writes into matches.status
UPCOMING_STATUSES = ["SCHEDULED", "TIMED"]
SETTLED_STATUSES = ["FINISHED", "POSTPONED", "CANCELLED", "AWARDED", "SUSPENDED"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FINISHED_SQL = text("""
    SELECT id, season, matchweek, kickoff_utc, home_team_id, away_team_id, home_goals, away_goals
    FROM matches
    WHERE status = 'FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL
    ORDER BY kickoff_utc ASC
""")

UPCOMING_SQL = text("""
    SELECT m.id, m.season, m.matchweek, m.kickoff_utc, m.home_team_id, m.away_team_id,
           ht.name AS home_name, awt.name AS away_name
    FROM matches m
    JOIN teams ht  ON m.home_team_id = ht.id
    JOIN teams awt ON m.away_team_id = awt.id
    WHERE m.status IN :statuses
      AND m.home_goals IS NULL
      AND m.kickoff_utc > NOW()
      AND m.kickoff_utc <= NOW() + (:hours * INTERVAL '1 hour')
      AND NOT EXISTS (
          SELECT 1 FROM predictions p
          WHERE p.match_id = m.id AND p.model_version = :model_version
      )
    ORDER BY m.kickoff_utc ASC
""").bindparams(bindparam("statuses", expanding=True))

MISSING_PREVIEWS_SQL = text("""
    SELECT p.match_id, p.p_home, p.p_draw, p.p_away,
           p.expected_home_goals, p.expected_away_goals,
           ht.name AS home_name, awt.name AS away_name
    FROM predictions p
    JOIN matches m ON m.id = p.match_id
    JOIN teams ht  ON m.home_team_id = ht.id
    JOIN teams awt ON m.away_team_id = awt.id
    LEFT JOIN match_previews mp
           ON mp.match_id = p.match_id AND mp.model_version = p.model_version
    WHERE p.model_version = :model_version
      AND mp.id IS NULL
      AND m.kickoff_utc > NOW() - INTERVAL '2 days'
    ORDER BY m.kickoff_utc ASC
    LIMIT :limit
""")

INSERT_PREDICTION_SQL = """
    INSERT INTO predictions (
        match_id, model_version, p_home, p_draw, p_away,
        expected_home_goals, expected_away_goals, scoreline_matrix,
        payload_hash, params, n_train, train_cutoff
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (match_id, model_version) DO NOTHING
"""

INSERT_PREVIEW_SQL = """
    INSERT INTO match_previews (match_id, model_version, text, llm_model)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (match_id, model_version) DO NOTHING
"""


# --------------------------------------------------------------------------- helpers
def code_sha():
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def canonical_hash(payload):
    """SHA-256 over deterministic, sorted, compact JSON."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_migration_applied(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
                WHERE c.relname = 'predictions' AND t.tgname = 'trg_predictions_lock_rules'
                  AND NOT t.tgisinternal
            )
        """)
        if not cur.fetchone()[0]:
            raise RuntimeError("Lock-rule trigger missing. Run: python -m src.migrations.upgrade_schema")
    conn.rollback()   # end the read-only transaction cleanly


def check_stale_ingest(conn):
    """Refuse to train if past matches were never settled (a failed ingest)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM matches
            WHERE kickoff_utc < NOW() - INTERVAL '4 hours'
              AND status <> ALL(%s)
            """,
            (SETTLED_STATUSES,),
        )
        unsettled = cur.fetchone()[0]
    conn.rollback()
    if unsettled > 0:
        raise RuntimeError(f"Found {unsettled} unsettled past matches. Ingest looks stale. Aborting.")


def league_average_rates(finished_df, reference, time_decay):
    """Decay-weighted league-average home / away goals: the 'no team information' baseline."""
    kickoff = pd.to_datetime(finished_df["kickoff_utc"], utc=True)
    days = ((reference - kickoff).dt.total_seconds() / 86400.0).clip(lower=0.0).to_numpy()
    weights = np.exp(-time_decay * days)
    lam = float(np.average(finished_df["home_goals"].to_numpy(dtype=float), weights=weights))
    mu = float(np.average(finished_df["away_goals"].to_numpy(dtype=float), weights=weights))
    return lam, mu


def validate_matrix(matrix, probs):
    if not np.isfinite(matrix).all():
        raise ValueError("Score matrix contains non-finite values")
    if abs(float(matrix.sum()) - 1.0) > 1e-9:
        raise ValueError(f"Score matrix sums to {matrix.sum()}, expected 1")
    derived = outcome_probs(matrix)
    if any(abs(a - b) > 1e-9 for a, b in zip(derived, probs)):
        raise ValueError("1X2 probabilities disagree with the matrix")
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("1X2 probabilities do not sum to 1")


def build_record(match_id, model_version, probs, xg, matrix, params, n_train, train_cutoff):
    payload = {
        "match_id": match_id,
        "model_version": model_version,
        "probabilities": list(probs),
        "expected_goals": list(xg),
        "matrix": matrix.tolist(),
        "params": params,
        "n_train": n_train,
        "train_cutoff": train_cutoff.isoformat(),
    }
    return payload, canonical_hash(payload)


def insert_prediction(cur, match_id, model_version, probs, xg, matrix, params, n_train,
                      train_cutoff, payload_hash):
    cur.execute(
        INSERT_PREDICTION_SQL,
        (match_id, model_version, probs[0], probs[1], probs[2], xg[0], xg[1],
         json.dumps(matrix.tolist()), payload_hash, json.dumps(params), n_train,
         train_cutoff.to_pydatetime()),
    )
    return cur.rowcount


# ------------------------------------------------------------------------ main steps
def lock_predictions(conn, engine, now_utc):
    """Train once, lock every eligible match. Returns (locked, skipped, errors)."""
    finished_df = pd.read_sql(FINISHED_SQL, engine)
    n_train = len(finished_df)
    if n_train < MIN_TRAIN_MATCHES:
        raise RuntimeError(f"Only {n_train} finished matches; refusing to train")
    train_cutoff = pd.to_datetime(finished_df["kickoff_utc"], utc=True).max()

    upcoming_df = pd.read_sql(
        UPCOMING_SQL, engine,
        params={"statuses": UPCOMING_STATUSES, "hours": LOCK_WINDOW_HOURS,
                "model_version": MODEL_VERSION},
    )
    if upcoming_df.empty:
        logging.info("No new matches entering the %d-hour lock window.", LOCK_WINDOW_HOURS)
        return 0, 0, 0

    logging.info("Training %s on %d matches...", MODEL_VERSION, n_train)
    model = DixonColesModel(time_decay=TIME_DECAY, ridge=RIDGE)
    model.fit(finished_df, reference_date=now_utc)
    params = {**model.describe(), "code_sha": code_sha(), "max_goals": MAX_GOALS}

    base_lam, base_mu = league_average_rates(finished_df, pd.Timestamp(now_utc), TIME_DECAY)
    base_matrix = score_matrix(base_lam, base_mu, 0.0, MAX_GOALS)
    base_probs = outcome_probs(base_matrix)
    base_params = {"kind": "league_average_poisson", "time_decay": TIME_DECAY,
                   "lambda": base_lam, "mu": base_mu, "n_matches": n_train,
                   "code_sha": params["code_sha"], "max_goals": MAX_GOALS}

    locked = skipped = errors = 0
    for match in upcoming_df.itertuples(index=False):
        match_id = int(match.id)
        label = f"match_id={match_id} ({match.home_name} vs {match.away_name})"
        try:
            p_home, p_draw, p_away, lam, mu, matrix = model.predict_probs(
                int(match.home_team_id), int(match.away_team_id), MAX_GOALS)
            probs = (p_home, p_draw, p_away)
            validate_matrix(matrix, probs)
            _, payload_hash = build_record(match_id, MODEL_VERSION, probs, (lam, mu), matrix,
                                           params, n_train, train_cutoff)
            _, base_hash = build_record(match_id, BASELINE_VERSION, base_probs,
                                        (base_lam, base_mu), base_matrix, base_params,
                                        n_train, train_cutoff)
            with conn.cursor() as cur:
                inserted = insert_prediction(cur, match_id, MODEL_VERSION, probs, (lam, mu),
                                             matrix, params, n_train, train_cutoff, payload_hash)
                insert_prediction(cur, match_id, BASELINE_VERSION, base_probs,
                                  (base_lam, base_mu), base_matrix, base_params, n_train,
                                  train_cutoff, base_hash)
            conn.commit()
            if inserted:
                locked += 1
                logging.info("Locked %s", label)
            else:
                logging.info("Already locked: %s", label)
        except UnknownTeamError as exc:
            conn.rollback()
            skipped += 1
            logging.warning("Skipped %s: no rating for team id %s", label, exc)
        except Exception:
            conn.rollback()
            errors += 1
            logging.exception("Error locking %s", label)
    return locked, skipped, errors


def generate_missing_previews(conn, engine):
    """Best effort. A failure here never affects a locked prediction."""
    try:
        from src.services.llm_analyst import LLMAnalyst, PreviewError
        analyst = LLMAnalyst()
    except Exception:
        logging.exception("LLM analyst unavailable; skipping previews")
        return 0
    if not analyst.available:
        logging.warning("GROQ_API_KEY is not set; skipping previews (the API shows templated text instead)")
        return 0

    pending = pd.read_sql(MISSING_PREVIEWS_SQL, engine,
                          params={"model_version": MODEL_VERSION, "limit": PREVIEW_BATCH_LIMIT})
    created = 0
    for row in pending.itertuples(index=False):
        try:
            preview = analyst.generate_match_preview(
                row.home_name, row.away_name,
                float(row.p_home), float(row.p_draw), float(row.p_away),
                float(row.expected_home_goals), float(row.expected_away_goals),
            )
            if not preview or not str(preview).strip():
                raise ValueError("empty preview")
            with conn.cursor() as cur:
                cur.execute(INSERT_PREVIEW_SQL,
                            (int(row.match_id), MODEL_VERSION, str(preview).strip(),
                             getattr(analyst, "model", None)))
            conn.commit()
            created += 1
        except PreviewError as exc:
            conn.rollback()
            logging.warning("Preview failed for match_id=%s (will retry next run): %s", row.match_id, exc)
        except Exception:
            conn.rollback()
            logging.exception("Preview failed for match_id=%s (will retry next run)", row.match_id)
    return created


def main():
    engine = get_sqlalchemy_engine()
    conn = get_db_connection()
    now_utc = pd.Timestamp.now(tz="UTC")
    try:
        assert_migration_applied(conn)
        check_stale_ingest(conn)
        locked, skipped, errors = lock_predictions(conn, engine, now_utc)
        logging.info("Locking complete: %d locked, %d skipped (unknown team), %d errors",
                     locked, skipped, errors)
        previews = generate_missing_previews(conn, engine)
        logging.info("Previews created: %d", previews)
        return 1 if errors else 0
    finally:
        conn.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())