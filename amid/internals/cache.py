from typing import Any, Callable, Sequence, Union

import numpy as np
from bev import Repository
from connectome import CacheColumns as Columns, CacheToDisk as Disk
from connectome.utils import StringsLike
from tarn import ReadError
from tarn.serializers import (
    ChainSerializer,
    ContentsIn,
    ContentsOut,
    DictSerializer,
    JsonSerializer as BaseJsonSerializer,
    NumpySerializer,
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
            cache.local,
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
            cache.local,
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


class JsonSerializer(BaseJsonSerializer):
    def save(self, value: Any, write: Callable) -> ContentsOut:
        # if namedtuple
        if isinstance(value, tuple) and hasattr(value, '_asdict') and hasattr(value, '_fields'):
            raise SerializerError

        return super().save(value, write)


class CleanInvalid(Serializer):
    def save(self, value: Any, write: Callable) -> ContentsOut:
        raise SerializerError

    def load(self, contents: ContentsIn, read: Callable) -> Any:
        raise ReadError
