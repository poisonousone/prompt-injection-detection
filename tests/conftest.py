"""Keep temporary test data in the workspace with inherited directory permissions."""

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(pytestconfig: pytest.Config) -> Iterator[Path]:
    # Python 3.14 owner-only temporary ACLs are incompatible with some Windows
    # restricted tokens. Ordinary workspace directories retain the parent's ACL.
    root = (pytestconfig.rootpath / ".cache" / "tests").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise ValueError("Unexpected test directory location")
        shutil.rmtree(path)
