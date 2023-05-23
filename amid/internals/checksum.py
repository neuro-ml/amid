import functools
import tempfile
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

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


def checksum(path: str, *, ignore=(), cache_columns=()):
    def make_cache():
        yield CacheToDisk(AntiSet(('id', *cache_columns)), serializer=serializer)
        if cache_columns:
            yield CacheColumns(cache_columns, serializer=serializer, verbose=True)

    serializer = default_serializer(None)

    def decorator(cls):
        class Checked(Chain):
            def __init__(self, root: str = None, version: str = Local):
                args = self._raw(root, version)

                if hasattr(cls, 'normalizer'):
                    args.append(cls.normalizer())
                    repository = get_repo(strict=False)
                    if repository is not None and repository.cache is not None and repository.cache.local:
                        args.extend(make_cache())

                self._version = version
                super().__init__(*args)

            @staticmethod
            def _raw(root: str = None, version: str = Local):
                ds = cls(root=root)
                args = [ds]

                if version is not None:
                    repository = get_repo(strict=False)
                    if repository is not None:
                        args.append(
                            CacheAndCheck(
                                set(dir(ds)) - {'id', 'ids', *ignore},
                                repository,
                                path,
                                fetch=True,
                                serializer=serializer,
                                version=version,
                            )
                        )

                return args

            @classmethod
            def raw(cls, root: str = None, version: str = Local):
                return Chain(*cls._raw(root, version))

            def _populate(
                self, *, ignore_errors: bool = False, cache: bool = True, fetch: bool = True, n_jobs: int = 1
            ):
                repository = get_repo()
                ds = self[0]
                fields = sorted(set(dir(ds)) - {'ids', 'id', *ignore})

                if cache:
                    ds = Chain(ds, *make_cache())
                ids = ds.ids

                ds = ds >> CacheAndCheck(
                    fields,
                    repository,
                    path,
                    fetch=fetch,
                    serializer=serializer,
                    version=self._version,
                    return_tree=True,
                )
                _loader = ds._compile(fields)

                def loader(key):
                    try:
                        return key, _loader(key)
                    except Exception as e:
                        if not ignore_errors:
                            raise RuntimeError(f'Error while processing id {key}') from e

                        return key, None

                print(f'Populating the cache with {len(fields)} fields for {len(ids)} ids')

                checksums = {}
                successes = errors = 0
                with ProgressParallel(n_jobs=n_jobs, backend='threading', tqdm_kwargs=dict(total=len(ids))) as bar:
                    for i, trees in tqdm(bar(map(delayed(loader), ids)), 'Saving the checksums'):
                        if trees is None:
                            errors += 1
                            continue

                        successes += 1
                        for name, tree in zip_equal(fields, trees):
                            for k, v in tree.items():
                                checksums['/'.join((name, i, k))] = v

                save_hash(checksums, to_hash(Path(repository.path / path)), repository.storage)
                return successes, errors

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
        fetch: bool = True,
        version,
    ):
        super().__init__(names, False)
        serializer = default_serializer(serializer)
        # name -> identifier -> tree
        checksums = defaultdict(lambda: defaultdict(dict))

        with suppress(HashNotFound, ReadError):
            for key, value in repository.load_tree(to_hash(Path(path)), version=version, fetch=fetch).items():
                name, identifier, relative = key.split('/', 2)
                if name in names:
                    checksums[name][identifier][relative] = value

        self.checksums = dict(checksums)
        self.return_tree = return_tree
        self.storage = repository.storage
        self.serializer = serializer
        self.keys = 'ids'

    def _get_storage(self) -> Cache:
        return self.storage

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

        for name in forward_outputs:
            if name in self.names:
                if not self.impure:
                    self._detect_impure(mapping[forward_outputs[name]], name)

                inp, out = Node(name, details), Node(name, details)
                inputs.append(inp)
                outputs.append(out)
                edges.append(
                    CheckSumEdge(
                        self.checksums.get(name, {}),
                        self.serializer,
                        self._get_storage(),
                        self.return_tree,
                        False,
                    ).bind([inp, key], out)
                )

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
    def __init__(self, tree, serializer, storage, return_tree: bool, check: bool):
        super().__init__(arity=2)
        self.return_tree = return_tree
        self._serializer, self._storage = serializer, storage
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

            value, exists = self._deserialize(expected)
            if exists:
                return value

        # 2. noting found - compute the value and save it in the storage
        value = yield Command.ParentValue, 0

        # check consistency
        if (expected is not None and self.check) or self.return_tree:
            # TODO: `check` is not the same as save to storage
            tree = self._serialize(value)
            if expected is not None and self.check and tree != expected:
                raise ValueError(f'Checksum failed for {identifier}. Actual: {tree}, expected: {expected}')
            if self.return_tree:
                return tree

        return value

    def _deserialize(self, tree):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            for k, v in tree.items():
                k = base / k
                k.parent.mkdir(parents=True, exist_ok=True)
                with open(k, 'w') as file:
                    file.write(v)

            try:
                return self._serializer.load(base, self._storage), True
            except ReadError as e:
                if isinstance(e, DeserializationError):
                    locations = {}
                    for k, v in tree.items():
                        try:
                            locations[k] = self._storage.read(lambda x: x, v)
                        except ReadError:
                            pass

                    raise DeserializationError(f'{tree}: {locations}')
                return None, False

    def _serialize(self, value):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            self._serializer.save(value, base)
            tree = {}
            # TODO: this is basically `mirror to storage`
            for file in base.glob('**/*'):
                if file.is_dir():
                    continue

                tree[str(file.relative_to(base))] = self._storage.write(file, labels=['amid.checksum']).hex()

            return tree


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
