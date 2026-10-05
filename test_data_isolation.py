"""The test suite must never write to the production data directory.

``build_rack_signal`` appends to ``prediction_log.csv`` via ``main.DATA_DIR``.
Five tests called it without redirecting that path, so running ``pytest``
locally added a fake ``live`` prediction row dated today to the system of
record -- a row that would then be committed by whoever ran the suite.

``conftest.py`` now redirects ``main.DATA_DIR`` to a per-test sandbox for every
test.  These assertions keep that in place.
"""

import csv
import os
from datetime import datetime

import main
import conftest


def test_main_data_dir_is_sandboxed_during_tests():
    """The root cause: a test must not see the production directory."""
    assert os.path.abspath(main.DATA_DIR) != os.path.abspath(conftest.REAL_DATA_DIR), (
        "main.DATA_DIR points at the production data directory during a test; "
        "anything that appends to the prediction log will corrupt it")


def test_sandbox_is_seeded_so_reads_still_work():
    """Isolation must not break tests that legitimately read real content."""
    assert os.path.exists(os.path.join(main.DATA_DIR, "graves_history.csv"))
    assert os.path.exists(os.path.join(main.DATA_DIR, "metrics_cache.json"))


def test_writes_land_in_the_sandbox_not_production(tmp_path):
    """Write through the real code path and prove production is untouched."""
    production = os.path.join(conftest.REAL_DATA_DIR, "prediction_log.csv")
    before = os.path.getsize(production) if os.path.exists(production) else None

    provenance = {"signal_contract": "/RBV26", "baseline_contract": "/RBV26",
                  "settlement_source": "schwab", "baseline_source": "schwab",
                  "settlement_captured_at": "", "status": "verified"}
    decision = main.decision_provenance("RB", 2.10, 2.00)
    main.append_prediction_log("RB", datetime.now(main.TZ), "HIKE", 10.0, 1.0,
                               provenance, decision)

    sandbox_log = os.path.join(main.DATA_DIR, "prediction_log.csv")
    assert os.path.exists(sandbox_log), "the write did not land in the sandbox"
    with open(sandbox_log, newline="") as handle:
        assert sum(1 for _ in csv.reader(handle)) >= 2

    after = os.path.getsize(production) if os.path.exists(production) else None
    assert before == after, "the production prediction log was modified by a test"


def test_the_opt_in_fixture_still_reaches_the_real_directory(real_data_dir):
    """Tests that genuinely need the real files can still ask for them."""
    assert os.path.abspath(main.DATA_DIR) == os.path.abspath(conftest.REAL_DATA_DIR)
    assert os.path.exists(os.path.join(main.DATA_DIR, "graves_history.csv"))


def test_production_log_has_no_row_dated_today_from_a_test_run():
    """A direct check on the artifact this defect used to damage.

    A genuine live row is written by the tracker at 2:35 PM CT with a real
    settlement source.  A row left by a test carries 'live_proxy:unknown' and
    an 'unavailable' contract status.
    """
    production = os.path.join(conftest.REAL_DATA_DIR, "prediction_log.csv")
    if not os.path.exists(production):
        return
    with open(production, newline="") as handle:
        rows = list(csv.DictReader(handle))

    suspicious = [
        row for row in rows
        if row.get("settlement_source") == "live_proxy:unknown"
        and row.get("contract_provenance_status") == "unavailable"
        and row.get("signal_contract") == "unknown"
    ]
    assert not suspicious, (
        f"{len(suspicious)} row(s) in the production prediction log look like "
        f"test artefacts, e.g. {suspicious[0]['timestamp']}")
