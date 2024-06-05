import contextlib
import zipfile
from pathlib import Path

import pandas as pd

from ..internals.dataset import register_field


def csv_field(name, cast):
    def _loader(self, i):
        value = self._meta[i].get(name)
        if pd.isnull(value):
            return None
        if cast is not None:
            return cast(value)
        return value

    register_field('RSNABreastCancer', name, _loader)
    return _loader


@contextlib.contextmanager
def unpack(root: str, relative: str):
    unpacked = Path(root) / relative

    if unpacked.exists():
        yield unpacked, True
    else:
        with zipfile.Path(root, relative).open('rb') as unpacked:
            yield unpacked, False
