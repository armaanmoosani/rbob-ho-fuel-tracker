"""Append-only, point-in-time calibration artifacts.

The artifact ledger is deliberately separate from metrics_cache.json.  The cache
is a live operational input that can legitimately change; an artifact records
the exact calibration that was eligible for one decision session.
"""

import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime


ARTIFACT_SCHEMA_VERSION = 2
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {1, 2}


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
    for index, artifact in enumerate(artifacts):
        expected_prior = "bootstrap_config" if index == 0 else artifacts[index - 1]["artifact_id"]
        if artifact["prior_artifact_id"] != expected_prior:
            raise ValueError(
                f"Calibration artifact chain is broken at {artifact['effective_session']}."
            )
        if index and artifact["training_end"] < artifacts[index - 1]["training_end"]:
            raise ValueError("Calibration training boundaries may not move backward.")
    return artifacts


def validate_calibration_artifact(artifact, line_number=None):
    required = {
        "artifact_schema_version", "artifact_id", "effective_session",
        "training_start", "training_end", "purge_rows", "source_history_hash",
        "source_row_count", "candidate_grid_version", "objective",
        "prior_artifact_id", "calibration", "generated_at",
    }
    schema_version = artifact.get("artifact_schema_version")
    if schema_version == 2:
        required.update({"candidate_grid", "smoothing_input"})
    missing = required.difference(artifact)
    label = f" at line {line_number}" if line_number else ""
    if missing:
        raise ValueError(f"Calibration artifact{label} is missing {sorted(missing)}.")
    if artifact["artifact_schema_version"] not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported calibration artifact schema{label}.")
    if artifact["purge_rows"] < 1 or artifact["source_row_count"] < 1:
        raise ValueError(f"Calibration artifact{label} has an invalid training boundary.")
    try:
        effective = date.fromisoformat(artifact["effective_session"])
        training_start = date.fromisoformat(artifact["training_start"])
        training_end = date.fromisoformat(artifact["training_end"])
        generated = datetime.fromisoformat(artifact["generated_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Calibration artifact{label} has an invalid date.") from exc
    if training_start > training_end or training_end >= effective:
        raise ValueError(f"Calibration artifact{label} has impossible date boundaries.")
    if generated.tzinfo is None:
        raise ValueError(f"Calibration artifact{label} generation time must include a timezone.")
    for field in ("artifact_id", "source_history_hash"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact[field])):
            raise ValueError(f"Calibration artifact{label} has an invalid {field}.")
    prior = str(artifact["prior_artifact_id"])
    if prior != "bootstrap_config" and not re.fullmatch(r"[0-9a-f]{64}", prior):
        raise ValueError(f"Calibration artifact{label} has an invalid prior artifact id.")
    if not isinstance(artifact["calibration"], dict) or not artifact["calibration"]:
        raise ValueError(f"Calibration artifact{label} has no calibration payload.")
    if artifact["artifact_schema_version"] >= 2:
        if not isinstance(artifact["candidate_grid"], dict) or not artifact["candidate_grid"]:
            raise ValueError(f"Calibration artifact{label} has no candidate grid.")
        if not isinstance(artifact["smoothing_input"], dict) or not artifact["smoothing_input"]:
            raise ValueError(f"Calibration artifact{label} has no smoothing input.")
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
    expected_prior = artifacts[-1]["artifact_id"] if artifacts else "bootstrap_config"
    if artifact["prior_artifact_id"] != expected_prior:
        raise ValueError("New calibration artifact does not continue the existing chain.")
    if artifacts and artifact["training_end"] < artifacts[-1]["training_end"]:
        raise ValueError("Calibration training boundaries may not move backward.")

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False,
            prefix=".calibration-runs-", suffix=".tmp",
        ) as handle:
            temp_path = handle.name
            for existing in artifacts:
                handle.write(canonical_json(existing) + "\n")
            handle.write(canonical_json(artifact) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    return artifact, True


def artifact_for_session(path, effective_session):
    session = str(effective_session)[:10]
    for artifact in load_calibration_artifacts(path):
        if artifact["effective_session"] == session:
            return artifact
    raise CalibrationArtifactUnavailable(
        f"No immutable calibration artifact exists for {session}."
    )
