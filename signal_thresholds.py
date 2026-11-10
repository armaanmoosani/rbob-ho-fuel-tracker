"""Baseline-dependent thresholds shared by live decisions and validation."""

CALIBRATED_BASELINE_SOURCES = frozenset({
    "settlement_provenance_verified",
    "graves_history_same_contract",
})


def effective_signal_thresholds(calibration, commodity, baseline_source):
    """Apply the decision-time fallback floor, keeping all values in cents/gal."""
    hike = calibration.get(f"{commodity}_HIKE_THRESHOLD_CENTS", 1.0)
    drop = calibration.get(f"{commodity}_DROP_THRESHOLD_CENTS", -1.0)
    lean_hike = calibration.get(f"{commodity}_LEAN_HIKE_CENTS", 0.5)
    lean_drop = calibration.get(f"{commodity}_LEAN_DROP_CENTS", -0.5)
    if baseline_source not in CALIBRATED_BASELINE_SOURCES:
        floor = calibration.get("FALLBACK_NOISE_FLOOR_CENTS", 1.2)
        hike = max(hike, floor)
        drop = min(drop, -floor)
        lean_hike = max(lean_hike, floor)
        lean_drop = min(lean_drop, -floor)
    return {
        "hike": hike,
        "drop": drop,
        "lean_hike": min(lean_hike, hike),
        "lean_drop": max(lean_drop, drop),
    }
