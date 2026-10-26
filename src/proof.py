"""
Canonical payload + SHA-256 hash for a locked prediction.

Used by BOTH the pipeline (to create the hash) and the API (to let anyone recompute
it), so the two can never drift apart. Do not change the payload shape for an existing
model_version: a change here is a new proof format.
"""
import hashlib
import json
from datetime import timezone

HASH_ALGORITHM = (
    "SHA-256 over the UTF-8 bytes of json.dumps(payload, sort_keys=True, "
    "separators=(',', ':'))"
)


def utc_iso(moment):
    """Timestamps are always hashed as UTC, whatever timezone the database session uses."""
    return moment.astimezone(timezone.utc).isoformat()


def build_payload(match_id, model_version, probs, xg, matrix_rows, params, n_train, train_cutoff):
    return {
        "match_id": int(match_id),
        "model_version": model_version,
        "probabilities": [float(p) for p in probs],
        "expected_goals": [float(v) for v in xg],
        "matrix": matrix_rows,
        "params": params,
        "n_train": int(n_train),
        "train_cutoff": utc_iso(train_cutoff),
    }


def canonical_hash(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()