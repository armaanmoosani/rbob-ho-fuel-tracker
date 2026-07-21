"""Append-only, point-in-time calibration artifacts.

The artifact ledger is deliberately separate from metrics_cache.json.  The cache
is a live operational input that can legitimately change; an artifact records
the exact calibration that was eligible for one decision session.
"""

import hashlib
import json
import os


ARTIFACT_SCHEMA_VERSION = 1


class CalibrationArtifactUnavailable(LookupError):
    """Raised when a requested session has no point-in-time calibration."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def artifact_id(artifact):
    """Hash the immutable content, excluding operational write metadata."""
    core = {key: value for key, value in artifact.items()
            if key not in {"artifact_id", "generated_at"}}
    return hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()


def load_calibration_artifacts(path):
    if not os.path.exists(path):
        return []

    artifacts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                artifact = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid calibration artifact at line {line_number}: {exc}") from exc
            validate_calibration_artifact(artifact, line_number)
            artifacts.append(artifact)

    sessions = [artifact["effective_session"] for artifact in artifacts]
    if sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
        raise ValueError("Calibration artifacts must have unique, chronological effective sessions.")
    return artifacts


def validate_calibration_artifact(artifact, line_number=None):
    required = {
        "artifact_schema_version", "artifact_id", "effective_session",
        "training_start", "training_end", "purge_rows", "source_history_hash",
        "source_row_count", "candidate_grid_version", "objective",
        "prior_artifact_id", "calibration", "generated_at",
    }
    missing = required.difference(artifact)
    label = f" at line {line_number}" if line_number else ""
    if missing:
        raise ValueError(f"Calibration artifact{label} is missing {sorted(missing)}.")
    if artifact["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration artifact schema{label}.")
    if artifact["purge_rows"] < 1 or artifact["source_row_count"] < 1:
        raise ValueError(f"Calibration artifact{label} has an invalid training boundary.")
    if not isinstance(artifact["calibration"], dict):
        raise ValueError(f"Calibration artifact{label} has no calibration payload.")
    expected = artifact_id(artifact)
    if artifact["artifact_id"] != expected:
        raise ValueError(f"Calibration artifact{label} has an invalid content hash.")


def append_calibration_artifact(path, artifact):
    """Append once, rejecting any attempt to change an existing session."""
    artifact = dict(artifact)
    artifact["artifact_id"] = artifact_id(artifact)
    validate_calibration_artifact(artifact)

    artifacts = load_calibration_artifacts(path)
    for existing in artifacts:
        if existing["effective_session"] == artifact["effective_session"]:
            if existing["artifact_id"] != artifact["artifact_id"]:
                raise ValueError(
                    "Refusing to replace the immutable calibration artifact for "
                    f"{artifact['effective_session']}."
                )
            return existing, False

    if artifacts and artifact["effective_session"] <= artifacts[-1]["effective_session"]:
        raise ValueError("Calibration artifacts may only be appended in session order.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(canonical_json(artifact) + "\n")
    return artifact, True


def artifact_for_session(path, effective_session):
    session = str(effective_session)[:10]
    for artifact in load_calibration_artifacts(path):
        if artifact["effective_session"] == session:
            return artifact
    raise CalibrationArtifactUnavailable(
        f"No immutable calibration artifact exists for {session}."
    )
