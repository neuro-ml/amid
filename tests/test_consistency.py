import pickle

import numpy as np
import pytest

from amid import AMOS
from amid.internals import gather_datasets

MAPPING = gather_datasets()
DATASETS = [x[0] for x in MAPPING.values()]
NAMES = list(MAPPING)
ROOT_MAPPING = {
    AMOS: '/shared/data/amos22',
}


@pytest.mark.parametrize('cls', DATASETS, ids=NAMES)
def test_ids_availability(cls):
    assert len(cls().ids) > 0


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


@pytest.mark.raw
@pytest.mark.parametrize('cls', ROOT_MAPPING, ids=[cls.__name__ for cls in ROOT_MAPPING])
def test_cache_consistency(cls):
    raw = cls(root=ROOT_MAPPING[cls])[0]
    cached = cls.raw()

    assert tuple(cached.ids) == tuple(raw.ids)
    cached_fields = set(dir(cached))
    raw_fields = set(dir(raw))
    assert cached_fields == raw_fields, (cached_fields - raw_fields, raw_fields - cached_fields)

    i = cached.ids[0]
    all_fields = cached_fields - {'ids'}
    compare(raw._compile(all_fields)(i), cached._compile(all_fields)(i))


# TODO: find a package for this
def compare(x, y):
    assert type(x) == type(y)
    if isinstance(x, (str, int, float, bytes)):
        assert x == y
    elif isinstance(x, np.ndarray):
        np.testing.assert_allclose(x, y)
    elif isinstance(x, (list, tuple)):
        list(map(compare, x, y))
    else:
        raise TypeError(type(x))
