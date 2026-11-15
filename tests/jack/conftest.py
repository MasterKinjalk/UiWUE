import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@pytest.fixture
def runs():
    import synthetic

    return synthetic.make_runs()


def point_core_at(root, monkeypatch):
    """Make uiwue.core (and everything built on it) read `root` as derived/."""
    from uiwue import core
    from uiwue.stats._design import clear_caches

    monkeypatch.setattr(core, "DERIVED", Path(root))
    clear_caches()


@pytest.fixture
def fake_derived(tmp_path, monkeypatch):
    """A synthetic derived/ folder that uiwue.core reads instead of the repo's."""
    import synthetic

    from uiwue.stats._design import clear_caches

    synthetic.write_derived(tmp_path)
    point_core_at(tmp_path, monkeypatch)
    yield tmp_path
    clear_caches()
