from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Return the local competition-data root used by integration tests."""
    root = Path("data/rogii-wellbore-geology-prediction")
    if not root.is_dir():
        pytest.fail(f"Competition data directory is missing: {root}")
    return root
