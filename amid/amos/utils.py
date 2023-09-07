from functools import partial

from connectome import Function

from .const import COLUMN2LABEL


def loader(column, i, _meta):
    # ambiguous data in meta
    if int(i) in [500, 600]:
        return None

    return _meta[_meta['amos_id'] == int(i)][column].item()


def add_labels(scope):
    for column, label in COLUMN2LABEL.items():
        scope[label] = Function(partial(loader, column), 'id', '_meta')
