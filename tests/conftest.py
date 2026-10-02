"""Tests never read the developer's .env, so they never call the hosted model."""

import pytest

from fastrfm import nvidia


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr(nvidia, "load_env", lambda: None)
    for name in nvidia.KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
