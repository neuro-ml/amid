from gzip import GzipFile
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
from bev import Repository
from connectome import CacheColumns as Columns, CacheToDisk as Disk
from connectome.utils import StringsLike
from tarn import HashKeyStorage, ReadError
from tarn.serializers import (
    ChainSerializer,
    DictSerializer,
    JsonSerializer as BaseJsonSerializer,
    PickleSerializer,
    Serializer,
    SerializerError,
)


class CacheToDisk(Disk):
    def __init__(
        self,
        names: StringsLike,
        serializer: Union[Serializer, Sequence[Serializer]] = None,
        **kwargs,
    ):
        repo = Repository.from_here('../data')
        cache = repo.cache
        super().__init__(
            [x.root for x in cache.local[0].locations],
            cache.storage,
            serializer=default_serializer(serializer),
            names=names,
            labels=['amid.cache'],
            **kwargs,
        )


class CacheColumns(Columns):
    def __init__(
        self,
        names: StringsLike,
        serializer: Union[Serializer, Sequence[Serializer]] = None,
        **kwargs,
    ):
        repo = Repository.from_here('../data')
        cache = repo.cache
        super().__init__(
            [x.root for x in cache.local[0].locations],
            cache.storage,
            serializer=default_serializer(serializer),
            names=names,
            labels=['amid.cache'],
            **kwargs,
        )


def default_serializer(serializers):
    if serializers is None:
        arrays = NumpySerializer({np.bool_: 1, np.integer: 1})
        serializers = ChainSerializer(
            JsonSerializer(),
            DictSerializer(serializer=arrays),
            arrays,
            PickleSerializer(),
            # CleanInvalid()
        )
    return serializers


class NumpySerializer(Serializer):
    def __init__(self, compression: Union[int, Dict[type, int]] = None):
        self.compression = compression

    def _choose_compression(self, value):
        if isinstance(self.compression, int) or self.compression is None:
            return self.compression

        if isinstance(self.compression, dict):
            for dtype in self.compression:
                if np.issubdtype(value.dtype, dtype):
                    return self.compression[dtype]

    def save(self, value, folder: Path):
        if not isinstance(value, (np.ndarray, np.generic)):
            raise SerializerError

        compression = self._choose_compression(value)
        if compression is not None:
            assert isinstance(compression, int)
            with GzipFile(folder / 'value.npy.gz', 'wb', compresslevel=compression, mtime=0) as file:
                np.save(file, value, allow_pickle=False)

        else:
            np.save(folder / 'value.npy', value, allow_pickle=False)

    def load(self, folder: Path, storage: HashKeyStorage):
        paths = list(folder.iterdir())
        if len(paths) != 1:
            raise SerializerError

        (path,) = paths
        if path.name == 'value.npy':
            loader = np.load
        elif path.name == 'value.npy.gz':

            def loader(x):
                with GzipFile(x, 'rb') as file:
                    return np.load(file)

        else:
            raise SerializerError

        try:
            return self._load_file(storage, loader, path)
        except (ValueError, EOFError) as e:
            raise ReadError from e


class JsonSerializer(BaseJsonSerializer):
    def save(self, value, folder: Path):
        # if namedtuple
        if isinstance(value, tuple) and hasattr(value, '_asdict') and hasattr(value, '_fields'):
            raise SerializerError

        return super().save(value, folder)


class CleanInvalid(Serializer):
    def save(self, value, folder: Path):
        raise SerializerError

    def load(self, folder: Path, storage: HashKeyStorage):
        raise ReadError
