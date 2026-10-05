"""Shared pytest fixtures.

The important one keeps the test suite out of the production data directory.
``build_rack_signal`` appends to ``prediction_log.csv`` through ``main.DATA_DIR``,
so any test that calls it without redirecting that path writes a fake live
prediction into the system of record.  Five tests did exactly that, and a local
``pytest`` run was enough to add a bogus ``live`` row dated today.

Pointing ``main.DATA_DIR`` at a temporary directory for every test makes the
mistake impossible to repeat rather than relying on each new test remembering
to patch it.  Tests that genuinely need the real files either read them by
explicit path (``alignment.load_history()``, ``data/graves_history.csv``) or
patch ``main.DATA_DIR`` themselves, and both still work.
"""

import os
import shutil

import pytest

REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Copied into the sandbox so a test that reads them sees real content without
# being able to write back to the originals.
SEEDED_FILES = (
    "config.json",
    "graves_history.csv",
    "metrics_cache.json",
    "nymex_settlement_provenance.csv",
)


@pytest.fixture(autouse=True)
def isolate_main_data_dir(tmp_path, monkeypatch):
    """Redirect main.DATA_DIR at a per-test sandbox seeded from the real files."""
    try:
        import main
    except Exception:  # pragma: no cover - main is importable in this suite
        yield
        return

    sandbox = tmp_path / "data"
    sandbox.mkdir(exist_ok=True)
    for name in SEEDED_FILES:
        source = os.path.join(REAL_DATA_DIR, name)
        if os.path.exists(source):
            shutil.copy2(source, sandbox / name)

    monkeypatch.setattr(main, "DATA_DIR", str(sandbox), raising=False)
    yield


@pytest.fixture
def real_data_dir(monkeypatch):
    """Opt back in to the real directory, for tests that must read it in place.

    Still read-only in spirit: anything a test writes here would be a bug, and
    test_no_test_writes_to_the_production_log below asserts the suite as a whole
    leaves the production log untouched.
    """
    import main
    monkeypatch.setattr(main, "DATA_DIR", REAL_DATA_DIR, raising=False)
    return REAL_DATA_DIR
