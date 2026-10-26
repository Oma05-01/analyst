from datetime import datetime, timedelta, timezone

from src.proof import build_payload, canonical_hash


def make(cutoff, probs=(0.5, 0.3, 0.2)):
    return build_payload(1, "model_x", probs, (1.5, 1.1), [[0.1, 0.9]], {"time_decay": 0.001}, 100, cutoff)


def test_hash_is_deterministic():
    cutoff = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    assert canonical_hash(make(cutoff)) == canonical_hash(make(cutoff))


def test_timezone_of_the_database_session_does_not_change_the_hash():
    utc = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    lagos = utc.astimezone(timezone(timedelta(hours=1)))
    assert canonical_hash(make(utc)) == canonical_hash(make(lagos))


def test_any_change_to_the_content_changes_the_hash():
    cutoff = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    assert canonical_hash(make(cutoff)) != canonical_hash(make(cutoff, probs=(0.5, 0.31, 0.19)))