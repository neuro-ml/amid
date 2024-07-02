from pathlib import Path
from typing import Sequence

from connectome import ExternalBase

from ..utils import PathOrStr


class Dataset(ExternalBase):
    _path: str
    _fields: Sequence[str] = None

    def __init__(self, root: PathOrStr):
        fields = None
        if hasattr(self, '_fields'):
            fields = self._fields

        super().__init__(fields=fields, inputs=['id'], properties=['ids'], inherit=['id'])
        self.root = Path(root)

    @classmethod
    def __getversion__(cls):
        return 0


_Fields = {}


def register_field(cls, name, func):
    _Fields.setdefault(cls, {})[name] = func


def field(func):
    cls, name = func.__qualname__.split('.')
    register_field(cls, name, func)
    return func
