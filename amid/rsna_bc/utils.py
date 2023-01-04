import contextlib
import zipfile
from pathlib import Path

import pandas as pd


fields = {
    'site_id': str,
    'patient_id': str,
    'image_id': str,
    'laterality': None,
    'view': None,
    'age': None,
    'cancer': bool,
    'biopsy': bool,
    'invasive': bool,
    'BIRADS': int,
    'implant': bool,
    'density': None,
    'machine_id': str,
    'prediction_id': str,
    'difficult_negative_case': bool,
}


def add_csv_fields(scope):
    def make_loader(field, cast):
        def loader(_row):
            value = _row.get(field)
            if pd.isnull(value):
                return None
            if cast is not None:
                return cast(value)
            return value

        return loader

    for _field, _cast in fields.items():
        scope[_field] = make_loader(_field, _cast)


@contextlib.contextmanager
def unpack(root: str, relative: str):
    unpacked = Path(root) / relative

    if unpacked.exists():
        yield unpacked, True
    else:
        with zipfile.Path(root, relative).open('rb') as unpacked:
            yield unpacked, False
