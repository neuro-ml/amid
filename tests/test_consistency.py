import pickle

import numpy as np
import pytest

from amid.internals import gather_datasets


MAPPING = gather_datasets()
DATASETS = [x[0] for x in MAPPING.values()]
NAMES = list(MAPPING)


@pytest.mark.raw
@pytest.mark.parametrize('cls', DATASETS, ids=NAMES)
def test_ids_availability(cls):
    assert len(cls().ids) > 0


@pytest.mark.raw
@pytest.mark.parametrize('cls', DATASETS, ids=NAMES)
def test_pickleable(cls):
    raw = cls()[0]
    cached = cls()
    fields = dir(raw)

    for ds in raw, cached:
        loader = ds._compile(fields)
        pickle.dumps(loader)

    f = cached._compile('ids')
    raw = pickle.dumps(f)
    g = pickle.loads(raw)
    assert f() == g()


# @pytest.mark.raw
# @pytest.mark.parametrize('cls', ROOT_MAPPING, ids=[cls.__name__ for cls in ROOT_MAPPING])
# def test_cache_consistency(cls):
#     raw = cls(root=ROOT_MAPPING[cls])
#     cached = raw.cached()
#     fields = {x.name for x in raw._container.outputs} - {'ids', 'id', 'cached'}
#
#     ids = raw.ids
#     assert ids == cached.ids
#     for i in ids:
#         for field in fields:
#             compare(getattr(raw, field)(i), getattr(cached, field)(i))


# TODO: find a package for this
def compare(x, y):
    assert type(x) == type(y)
    if isinstance(x, (str, int, float, bytes)):
        assert x == y
    elif isinstance(x, (np.ndarray, np.generic)):
        np.testing.assert_allclose(x, y)
    elif isinstance(x, (list, tuple)):
        list(map(compare, x, y))
    else:
        raise TypeError(type(x))
