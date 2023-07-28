import functools
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, Optional, Tuple

import numpy as np
from bev import Local, Repository
from bev.exceptions import HashNotFound
from bev.hash import to_hash
from bev.ops import save_hash
from connectome import Chain
from connectome.cache import Cache
from connectome.containers import EdgesBag, IdentityContext
from connectome.engine import Command, ConstantEdge, Details, Node, StaticGraph, StaticHash, TreeNode
from connectome.layers.cache import CacheToStorage
from connectome.utils import AntiSet, node_to_dict
from joblib import Parallel, delayed
from more_itertools import zip_equal
from tarn import DeserializationError, ReadError
from tqdm.auto import tqdm

from .base import get_repo
from .cache import CacheColumns, CacheToDisk, default_serializer


# TODO: add possibility to check the entire tree without the need to pull anything from remote


def checksum(path: str, *, ignore=(), columns=()):
    def _cache(cols=False):
        repository = get_repo(strict=False)
        if repository is not None and repository.cache is not None and repository.cache.local:
            yield CacheToDisk(AntiSet(('id', *columns)), serializer=serializer)
            if cols:
                yield CacheColumns(columns, serializer=serializer, verbose=True, shard_size=500)

    def _checker(ds, version):
        if version is not None:
            repository = get_repo(strict=False)
            if repository is not None:
                yield CacheAndCheck(
                    set(dir(ds)) - {'id', 'ids', *ignore},
                    repository.copy(version=version, fetch=True),
                    path,
                    serializer=serializer,
                )

    serializer = default_serializer(None)

    def decorator(cls):
        class Checked(Chain):
            def __init__(self, root: Optional[str] = None, version: str = Local):
                ds = cls(root=root)

                if hasattr(cls, 'normalizer'):
                    args = [
                        ds,
                        *_checker(ds, version),
                        cls.normalizer(),
                        *_cache(),
                    ]
                else:
                    args = [
                        ds,
                        *_cache(),
                        *_checker(ds, version),
                    ]

                self._version = version
                super().__init__(*args)

            @classmethod
            def raw(cls, root: Optional[str] = None, version: str = Local):
                ds = cls(root=root)
                return Chain(
                    ds,
                    *_cache(),
                    *_checker(ds, version),
                )

            def _populate(
                self,
                *,
                ignore_errors: bool = False,
                cache: bool = True,
                fetch: bool = True,
                n_jobs: int = 1,
                analyze_fields: bool = False,
            ):
                repository = get_repo().copy(fetch=fetch, version=Local)
                ds = self[0]
                fields = sorted(set(dir(ds)) - {'ids', 'id', *ignore})

                if cache:
                    ds = Chain(ds, *_cache(False))
                ids = sorted(ds.ids)

                checked = ds >> CacheAndCheck(
                    fields,
                    repository,
                    path,
                    serializer=serializer,
                    return_tree=True,
                )
                _loader = checked._compile(fields)

                def loader(key):
                    try:
                        return key, _loader(key)
                    except Exception as e:
                        if not ignore_errors:
                            raise RuntimeError(f'Error while processing id {key}') from e

                        return key, None

                print(f'Populating the cache with {len(fields)} fields for {len(ids)} ids')

                checksums = {}
                successes, errors = [], []
                hashes = defaultdict(list)
                with ProgressParallel(
                    n_jobs=n_jobs, backend='threading', tqdm_kwargs=dict(total=len(ids), desc='Populating checksums')
                ) as bar:
                    for i, trees in bar(map(delayed(loader), ids)):
                        if trees is None:
                            errors.append(i)
                            continue

                        successes.append(i)
                        for name, tree in zip_equal(fields, trees):
                            for k, v in tree.items():
                                hashes[name].append(v)
                                checksums['/'.join((name, i, k))] = v

                # we save the checksums 2 times to make sure the work isn't lost
                save_hash(checksums, to_hash(Path(repository.path / path)), repository.storage)

                # check the columns and give recommendations
                if analyze_fields:
                    #   we allow to cache to mem all fields that take <500mb
                    sizes = {}
                    for name, vs in tqdm(hashes.items(), 'Analyzing the fields'):
                        max_count = 50
                        mul = 1
                        if len(vs) > max_count:
                            mul = len(vs) / max_count
                            vs = np.random.choice(vs, max_count, replace=False)

                        sizes[name] = (
                            sum(repository.storage.read(lambda x: get_value_size(x) / 1024**2, v) for v in vs) * mul
                        )

                    add = {k for k, v in sizes.items() if v <= 500} - set(columns)
                    remove = {k for k, v in sizes.items() if v > 500} & set(columns)
                    if add or remove:
                        print(
                            f'It is recommended to add these fields to columns cache: {list(add)!r}, '
                            f'also, remove these fields: {list(remove)!r}, like so:\n'
                            f'@checksum(..., columns={sorted(add | set(columns))!r})'
                        )

                # build columns cache
                if columns:
                    values = defaultdict(list)
                    checked = Chain(ds, *_checker(ds, Local))
                    new_ids = sorted(checked.ids)
                    with ProgressParallel(
                        n_jobs=n_jobs,
                        backend='threading',
                        tqdm_kwargs=dict(total=len(new_ids), desc='Populating lightweight columns'),
                    ) as bar:
                        for vals in bar(map(delayed(checked._compile(columns)), new_ids)):
                            for k, v in zip_equal(columns, vals):
                                values[k].append(v)

                    for name, vals in values.items():
                        for k, v in serialize(vals, serializer, repository).items():
                            checksums['/'.join((f'_{name}', k))] = v

                    save_hash(checksums, to_hash(Path(repository.path / path)), repository.storage)
                return len(successes), len(errors)

        functools.update_wrapper(Checked, cls, updated=())
        return Checked

    return decorator


class CacheAndCheck(CacheToStorage):
    def __init__(
        self,
        names,
        repository: Repository,
        path,
        *,
        serializer=None,
        return_tree: bool = False,
    ):
        super().__init__(names, False)
        serializer = default_serializer(serializer)
        # name -> identifier -> tree
        checksums = defaultdict(lambda: defaultdict(dict))
        columns = defaultdict(dict)

        with suppress(HashNotFound, ReadError):
            for key, value in repository.load_tree(to_hash(Path(path))).items():
                name, relative = key.split('/', 1)
                if name.startswith('_'):
                    columns[name[1:]][relative] = value

                else:
                    identifier, relative = relative.split('/', 1)
                    if name in names:
                        checksums[name][identifier][relative] = value

        self.checksums = dict(checksums)
        self.columns = dict(columns)
        self.return_tree = return_tree
        self.repository = repository
        self.serializer = serializer
        self.keys = 'ids'

    def _get_storage(self) -> Cache:
        return self.repository

    def _prepare_container(self, previous: EdgesBag) -> EdgesBag:
        if len(previous.inputs) != 1:
            raise ValueError('The input layer must contain exactly one input')
        details = Details(type(self))
        key = Node(previous.inputs[0].name, details)

        mapping = TreeNode.from_edges(previous.edges)
        forward_outputs = node_to_dict(previous.outputs)

        inputs, outputs, edges = [key], [], []
        # do we need to add a `keys` output:
        if self.checksums or self.keys not in forward_outputs:
            forward_outputs.pop(self.keys, None)
            keys = Node(self.keys, details)
            outputs.append(keys)
            # TODO: what if the ids differ in checksum?
            edges.append(ConstantEdge(tuple(sorted(list(self.checksums.values())[0]))).bind((), keys))
        else:
            keys = forward_outputs[self.keys]

        for name in forward_outputs:
            if name in self.names:
                if not self.impure:
                    self._detect_impure(mapping[forward_outputs[name]], name)

                inp, out = Node(name, details), Node(name, details)
                inputs.append(inp)
                edges.append(
                    CheckSumEdge(
                        self.checksums.get(name, {}),
                        self.serializer,
                        self.repository,
                        self.return_tree,
                        False,
                    ).bind([inp, key], out)
                )
                if not self.return_tree and name in self.columns:
                    cached = Node(name, details)
                    edges.append(
                        CheckSumColumn(self.columns[name], self.serializer, self.repository).bind(
                            [out, key, keys], cached
                        )
                    )
                    outputs.append(cached)
                else:
                    outputs.append(out)

        return EdgesBag(
            inputs,
            outputs,
            edges,
            IdentityContext(),
            persistent=None,
            virtual=AntiSet(node_to_dict(outputs)),
            optional=None,
        )


class CheckSumEdge(StaticGraph, StaticHash):
    def __init__(self, tree, serializer, repository, return_tree: bool, check: bool):
        super().__init__(arity=2)
        self.return_tree = return_tree
        self._serializer, self._repository = serializer, repository
        self.tree = tree
        self.check = check

    def _make_hash(self, inputs):
        if self.return_tree:
            raise ValueError('In `return tree` mode graph hashing is not supported')
        return inputs[0]

    def evaluate(self):
        identifier = yield Command.ParentValue, 1
        expected = self.tree.get(identifier)

        # 1. try to deserialize directly from the storage
        if expected is not None:
            if self.return_tree:
                return expected

            value, exists = deserialize(expected, self._serializer, self._repository)
            if exists:
                return value

        # 2. noting found - compute the value and save it in the storage
        value = yield Command.ParentValue, 0

        # check consistency
        if (expected is not None and self.check) or self.return_tree:
            # TODO: `check` is not the same as save to storage
            tree = serialize(value, self._serializer, self._repository)
            if expected is not None and self.check and tree != expected:
                raise ValueError(f'Checksum failed for {identifier}. Actual: {tree}, expected: {expected}')
            if self.return_tree:
                return tree

        return value


class CheckSumColumn(StaticGraph, StaticHash):
    def __init__(self, tree, serializer, repository):
        super().__init__(arity=3)
        self._serializer, self._repository = serializer, repository
        self.tree = tree
        # FIXME
        self._cache = None

    def _make_hash(self, inputs):
        return inputs[0]

    def evaluate(self):
        if self._cache is None:
            keys = yield Command.ParentValue, 2
            values, success = deserialize(self.tree, self._serializer, self._repository)
            if success:
                self._cache = dict(zip_equal(keys, values))
            else:
                self._cache = {}

        key = yield Command.ParentValue, 1
        if key in self._cache:
            return self._cache[key]

        value = yield Command.ParentValue, 0
        return value


def deserialize(tree, serializer, repository) -> Tuple[Any, bool]:
    def read(fn, x):
        return repository.storage.read(fn, x, fetch=repository.fetch)

    try:
        return serializer.load(list(tree.items()), read), True

    except ReadError as e:
        if isinstance(e, DeserializationError):
            locations = {}
            for k, v in tree.items():
                try:
                    locations[k] = read(lambda x: x, v)
                except ReadError:
                    pass

            raise DeserializationError(f'{tree}: {locations}')
        return None, False


def serialize(value, serializer, repository):
    return dict(serializer.save(value, lambda v: repository.storage.write(v, labels=['amid.checksum']).hex()))


def get_value_size(x):
    if isinstance(x, (Path, str)):
        return Path(x).stat().st_size
    assert isinstance(x, BinaryIO)
    chunk_size, size = 2**16, 0
    value = x.read(chunk_size)
    while len(value) > 0:
        size += len(value)
        value = x.read(chunk_size)
    return size


# source: https://stackoverflow.com/a/61027781
class ProgressParallel(Parallel):
    def __init__(self, *args, tqdm_kwargs=None, **kwargs):
        self._tqdm_kwargs = tqdm_kwargs or {}
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(**self._tqdm_kwargs) as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        if 'total' not in self._tqdm_kwargs:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

    def __getattr__(self, name):
        return getattr(self._pbar, name)
