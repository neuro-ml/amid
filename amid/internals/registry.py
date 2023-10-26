import importlib
import inspect
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Type

import pandas as pd

from .checksum import checksum
from .licenses import License


_REGISTRY = {}


class Description(NamedTuple):
    body_region: str = None
    license: str = None
    link: str = None
    modality: str = None
    prep_data_size: str = None
    raw_data_size: str = None
    task: str = None


def register(**kwargs):
    def decorator(cls: Type):
        _register(cls, cls.__origin__.__name__, description, 2)
        return cls

    description = Description(**kwargs)
    return decorator


def normalize(cls: Type, name: str, short_name: str, *, normalizers=(), ignore=(), columns=(), **kwargs):
    cls = checksum(short_name, normalizers=normalizers, ignore=ignore, columns=columns)(cls)
    cls.__qualname__ = name
    description = Description(**kwargs)
    _register(cls, name, description, 2)
    return cls


def _register(cls, name, description, level):
    module = inspect.getmodule(inspect.stack()[level][0]).__name__
    assert name not in _REGISTRY, name
    _REGISTRY[name] = cls, module, description


def gather_datasets():
    for f in Path(__file__).resolve().parent.parent.iterdir():
        module_name = f'amid.{f.stem}'
        importlib.import_module(module_name)

    return OrderedDict((k, _REGISTRY[k]) for k in sorted(_REGISTRY))


def prepare_for_table(name, cls, module, description, version):
    def stringify(x):
        if pd.isnull(x):
            return ''
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            return ', '.join(x)
        return x

    entry = {'name': name, 'entries': len(cls().ids)}
    entry.update({k: v for k, v in description._asdict().items() if not pd.isnull(v)})
    license_ = entry.get('license', None)
    if license_:
        if isinstance(license_, License):
            license_ = f'<a href="{license_.url}">{license_.name}</a>'
        entry['license'] = license_

    link = entry.pop('link', None)
    if link is not None:
        entry['link'] = f'<a href="{link}">Source</a>'

    entry['name'] = f'<a href="https://neuro-ml.github.io/amid/{version}/datasets-api/#{module}.{name}">{name}</a>'
    return {k: stringify(v) for k, v in entry.items()}
