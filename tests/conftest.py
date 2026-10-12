"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _private_render_cache(tmp_path_factory):
    """Keep compiled test pictures out of the user's ~/.cache/pytikz."""
    patch = pytest.MonkeyPatch()
    patch.setenv("PYTIKZ_CACHE", str(tmp_path_factory.mktemp("pytikz-cache")))
    yield
    patch.undo()
