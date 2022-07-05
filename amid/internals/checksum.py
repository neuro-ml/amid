import tempfile
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

from bev import Repository, Local
from bev.cli.add import save_tree
from bev.exceptions import HashNotFound
from bev.hash import to_hash
from connectome import Chain
from connectome.containers.base import EdgesBag
from connectome.containers.cache import CacheContainer, IdentityContext
from connectome.engine.base import Command, TreeNode, Node
from connectome.engine.edges import StaticGraph, StaticHash, IdentityEdge, ConstantEdge
from connectome.interface.blocks import CacheLayer
from connectome.utils import node_to_dict, AntiSet
from more_itertools import zip_equal
from tarn import ReadError
from tqdm.auto import tqdm

from .base import get_repo
from .cache import default_serializer, CacheToDisk


# TODO: this file is a mess, but most of this logic will be moved
#  to connectome, bev and tarn eventually

# TODO: add possibility to check the entire tree without the need to pull anything from remote


def checksum(path: str):
    serializer = default_serializer(None)

    def decorator(cls):
        class Checked(Chain):
            def __init__(self, root: str = None, version: str = Local):
                ds = cls(root=root)
                args = [ds]

                if version is not None:
                    repository = get_repo(strict=False)
                    if repository is not None:
                        args.extend((
                            CacheToDisk(AntiSet(('id',)), serializer=serializer),
                            CacheAndCheck(
                                set(dir(ds)) - {'id', 'ids'}, repository, path, fetch=True,
                                serializer=serializer, version=version
                            ),
                        ))

                self._version = version
                super().__init__(*args)

            def _populate(self):
                repository = get_repo()
                ids = self.ids

                fields = sorted(set(dir(self[0])) - {'ids', 'id'})
                ds = self[0] >> CacheToDisk(AntiSet(('id',)), serializer=serializer) >> CacheAndCheck(
                    fields, repository, path, fetch=True,
                    serializer=serializer, version=self._version, return_tree=True,
                )
                loader = ds._compile(fields)

                checksums = {}
                with tqdm(ids, 'Populating the cache') as bar:
                    for i in bar:
                        bar.set_postfix_str(i)

                        try:
                            trees = loader(i)
                        except Exception as e:
                            print('Error for', i, type(e).__name__, e)
                            continue

                        for name, tree in zip_equal(fields, trees):
                            for k, v in tree.items():
                                checksums['/'.join((name, i, k))] = v

                save_tree(repository, checksums, to_hash(Path(repository.path / path)))

        return Checked

    return decorator


class CacheAndCheck(CacheLayer):
    def __init__(self, names, repository: Repository, path, *, serializer=None, return_tree: bool = False,
                 fetch: bool = True, version):
        serializer = default_serializer(serializer)
        # name -> identifier -> tree
        checksums = defaultdict(lambda: defaultdict(dict))

        with suppress(HashNotFound):
            for key, value in repository.load_tree(to_hash(Path(path)), version=version, fetch=fetch).items():
                name, identifier, relative = key.split('/', 2)
                if name in names:
                    checksums[name][identifier][relative] = value

        checksums = dict(checksums)
        super().__init__(CacheAndCheckContainer(names, serializer, repository.storage, checksums, return_tree))


class CacheAndCheckContainer(CacheContainer):
    def __init__(self, names, serializer, storage, checksums: dict, return_tree: bool):
        super().__init__(names, False)
        self.return_tree = return_tree
        self.checksums = checksums
        self.storage = storage
        self.serializer = serializer
        self.keys = 'ids'

    def get_storage(self):
        return self.storage

    def wrap(self, container: EdgesBag) -> EdgesBag:
        state = container.freeze()
        if len(state.inputs) != 1:
            raise ValueError('The input layer must contain exactly one input')
        key, = state.inputs
        edges = list(state.edges)
        forward_outputs = node_to_dict(state.outputs)
        outputs = []

        # do we need to add a `keys` output:
        if self.checksums or self.keys not in forward_outputs:
            forward_outputs.pop(self.keys, None)
            keys = Node(self.keys)
            outputs.append(keys)
            # TODO: what if the ids differ in checksum?
            edges.append(ConstantEdge(tuple(sorted(list(self.checksums.values())[0]))).bind((), keys))

        mapping = TreeNode.from_edges(edges)

        for node_name in forward_outputs:
            node = Node(node_name)
            outputs.append(node)

            output = forward_outputs[node_name]
            if self.cache_names is None or node_name in self.cache_names:
                if not self.allow_impure:
                    self._detect_impure(mapping[output], node_name)
                edges.append(CheckSumEdge(
                    self.checksums.get(node_name, {}), self.serializer, self.get_storage(), self.return_tree, True,
                ).bind([output, key], node))
            else:
                edges.append(IdentityEdge().bind(output, node))

        return EdgesBag([key], outputs, edges, IdentityContext(), persistent_nodes=state.persistent_nodes)


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
                raise ValueError(
                    f'Checksum failed for {identifier}. Actual: {tree}, expected: {expected}'
                )
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
            except ReadError:
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

                tree[str(file.relative_to(base))] = self._storage.write(file)

            return tree
