"""Execute the repo's notebooks end-to-end as tests.

Each notebook runs top-to-bottom in its own kernel via nbclient; any raised
cell fails the test. ponytail: kernels resolve from each notebook's pinned
kernelspec (replenishment-uv -> this repo's .venv), which is machine-local --
switch to a programmatically-registered kernel if this ever needs to run in CI.
"""
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NB_DIR = Path(__file__).parent.parent / "notebooks"


def _execute(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600).execute(cwd=str(path.parent))


def test_sbd_poc3_dummy_data_notebook_runs():
    _execute(NB_DIR / "sbd_poc3_dummy_data.ipynb")


def test_pourri_head_to_head_notebook_runs():
    if not (NB_DIR / "pourri" / "pourri_monthly.csv").exists():
        pytest.skip("pourri_monthly.csv not present -- re-run extraction.sql")
    _execute(NB_DIR / "pourri" / "pourri_head_to_head.ipynb")


def test_sbd_poc3_local_fixture_notebook_runs():
    fixture_dir = NB_DIR.parent / "fixtures" / "sbd_poc3"
    if not any(fixture_dir.glob("sbd_poc3.parquet")) and not any(fixture_dir.glob("sbd_poc3.csv")):
        pytest.skip("no SBD fixture file yet -- see fixtures/sbd_poc3/README.md")
    _execute(NB_DIR / "sbd_poc3_local_fixture.ipynb")
