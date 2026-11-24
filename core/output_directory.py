"""Publish a complete run to a new directory."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def new_output_directory(destination: Path):
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".staging-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "output"
        staging.mkdir()
        yield staging
        staging.rename(destination)
