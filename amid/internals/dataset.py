from pathlib import Path

from bev.utils import PathOrStr
from connectome import ExternalBase

from .checksum import Checked


class Dataset(ExternalBase):
    _path: str

    def __init__(self, root: PathOrStr):
        super().__init__(inputs=['id'], inherit=['id'])
        self.root = Path(root)

    @classmethod
    def __getversion__(cls):
        return 0

    def cached(self):
        return Checked(self, self._path)
