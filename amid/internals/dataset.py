from pathlib import Path
from typing import Sequence

from bev.utils import PathOrStr
from connectome import ExternalBase


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
