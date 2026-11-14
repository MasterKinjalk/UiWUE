import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def runs():
    import synthetic

    return synthetic.make_runs()


@pytest.fixture
def fake_derived(tmp_path, monkeypatch):
    """A synthetic `derived/` folder, pointed to by UIWUE_DERIVED."""
    import synthetic

    from uiwue.stats._design import _catalogue_cached

    synthetic.write_derived(tmp_path)
    monkeypatch.setenv("UIWUE_DERIVED", str(tmp_path))
    _catalogue_cached.cache_clear()
    yield tmp_path
    _catalogue_cached.cache_clear()
