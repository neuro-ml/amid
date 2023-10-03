import contextlib
import zipfile
from functools import partial
from pathlib import Path

import pandas as pd
from connectome import Positional


def _loader(cast, name, _row):
    value = _row.get(name)
    if pd.isnull(value):
        return None
    if cast is not None:
        return cast(value)
    return value


def csv_field(cast):
    return Positional(partial(_loader, cast), 'name', '_row')


@contextlib.contextmanager
def unpack(root: str, relative: str):
    unpacked = Path(root) / relative

    if unpacked.exists():
        yield unpacked, True
    else:
        with zipfile.Path(root, relative).open('rb') as unpacked:
            yield unpacked, False
