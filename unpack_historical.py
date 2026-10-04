"""Safely extract the historical scientific artifacts into their analysis paths."""
from pathlib import Path
from zipfile import ZipFile

root = Path(__file__).resolve().parent
with ZipFile(root / 'historical_results.zip') as archive:
    assert archive.testzip() is None
    for name in archive.namelist():
        assert (root / name).resolve().is_relative_to(root.resolve()), name
    archive.extractall(root)
print('Historical artifacts extracted; original stage paths preserved.')
