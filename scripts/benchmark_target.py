"""Discovery endpoint gate for controller-owned, independently checked evidence.

Call only after the frozen external oracle and its independent replay finish on
the exact immutable snapshot. Never pass model-written receipts to this gate.
Task-specific thresholds belong in the frozen protocol, not in app behavior.
"""
import gzip
import hashlib
import json
from pathlib import Path


def word_digest(path):
    word = json.loads(gzip.decompress(Path(path).read_bytes()))
    # A different gzip header or JSON formatting is not a different circuit.
    canonical = json.dumps(word, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def discovery_outcome(protocol, baseline_words, candidate_words, original, replay):
    expected = [stage["id"] for stage in protocol["milestones"]]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("A nonempty, unique milestone contract is required")
    if set(baseline_words) != set(candidate_words) or not baseline_words:
        raise ValueError("Candidate axes must match the frozen baseline")
    unchanged = all(word_digest(baseline_words[h]) == word_digest(candidate_words[h])
                    for h in baseline_words)
    if unchanged:
        return {"research_success": False, "classification": "baseline_reproduction"}
    if original.get("valid") is not True or replay.get("valid") is not True:
        return {"research_success": False, "classification": "invalid_or_unverified"}
    mathematical_fields = ("bit", "finite_bridge", "milestones", "replay")
    if any(key not in original or key not in replay or original[key] != replay[key]
           for key in mathematical_fields):
        return {"research_success": False, "classification": "independent_replay_mismatch"}
    for receipt in (original, replay):
        stages = receipt.get("milestones", [])
        if ([stage.get("id") for stage in stages] != expected or
                receipt.get("target_reached") is not True or
                not all(stage.get("target_reached") is True for stage in stages)):
            return {"research_success": False, "classification": "target_not_reached"}
    return {"research_success": True, "classification": "all_frozen_targets_reached"}
