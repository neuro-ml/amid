import inspect
from typing import NamedTuple, Type
import importlib
from collections import OrderedDict
from pathlib import Path

_REGISTRY = {}


class Description(NamedTuple):
    body_region: str = None
    modality: str = None
    task: str = None
    licence: str = None


def register(**kwargs):
    def decorator(cls: Type):
        name = cls.__name__
        module = inspect.getmodule(inspect.stack()[1][0]).__name__
        assert name not in _REGISTRY, name
        _REGISTRY[name] = cls, module, Description(**kwargs)
        return cls

    return decorator


def gather_datasets():
    for f in Path(__file__).resolve().parent.parent.iterdir():
        module_name = f'amid.{f.stem}'
        importlib.import_module(module_name)

    return OrderedDict((k, _REGISTRY[k]) for k in sorted(_REGISTRY))
