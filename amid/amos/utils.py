from functools import partial

from connectome import Function


def loader(column, i, _meta):
    # ambiguous data in meta
    if int(i) in [500, 600]:
        return None
    elif int(i) not in _meta['amos_id']:
        return None

    return _meta[_meta['amos_id'] == int(i)][column].item()


def label(column):
    return Function(partial(loader, column), 'id', '_meta')
