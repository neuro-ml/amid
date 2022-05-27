from collections import defaultdict

import numpy as np
from bev import Repository, Local
from bev.config import wrap_levels
from connectome.containers.base import EdgesBag
from connectome.containers.cache import CacheContainer, IdentityContext
from connectome.engine.base import Command, TreeNode, Node
from connectome.engine.edges import StaticGraph, StaticHash, IdentityEdge
from connectome.interface.blocks import CacheLayer
from connectome.utils import node_to_dict
from tarn.cache import CacheIndex, CacheStorage, NumpySerializer, JsonSerializer, DictSerializer, ChainSerializer
from tarn.cache.index import DATA_FOLDER
from tarn.digest import key_to_relative


# TODO: this file is a mess, but most of this logic will be moved
#  to connectome and tarn eventually

# TODO: add possibility to check the entire tree without the need to pull anything from remote

class CacheAndCheck(CacheLayer):
    def __init__(self, names, prefix, serializer=None, return_tree: bool = False):
        serializer = _default_serializer(serializer)
        # TODO: make the repo a global variable?
        repo = Repository.from_here('../data', version=Local)

        local, remote = repo.cache
        checksums = {}
        for file in (repo.root / prefix).glob('*.hash'):
            entry = defaultdict(dict)
            for key, value in repo.load_tree(file.relative_to(repo.root)).items():
                identifier, relative = key.split('/', 1)
                entry[identifier][relative] = value

            checksums[file.stem] = dict(entry)

        super().__init__(CacheAndCheckContainer(
            names, CacheStorage(
                *wrap_levels(local, CheckSumIndex, storage=repo.storage, serializer=serializer), remote=remote
            ), checksums, return_tree
        ))


def _default_serializer(serializers):
    if serializers is None:
        arrays = NumpySerializer({np.bool_: 1, np.int_: 1})
        serializers = ChainSerializer(
            JsonSerializer(),
            DictSerializer(serializer=arrays),
            arrays,
        )
    return serializers


class CacheAndCheckContainer(CacheContainer):
    def __init__(self, names, storage, checksums, return_tree: bool):
        super().__init__(names, False)
        self.return_tree = return_tree
        self.checksums = checksums
        self.storage = storage

    def get_storage(self):
        return self.storage

    def wrap(self, container: EdgesBag) -> EdgesBag:
        state = container.freeze()
        forward_outputs = node_to_dict(state.outputs)
        if len(state.inputs) != 1:
            raise ValueError('The input layer must contain exactly one input')
        key, = state.inputs

        edges = list(state.edges)
        outputs = [Node(name) for name in forward_outputs]
        mapping = TreeNode.from_edges(state.edges)

        for node in outputs:
            node_name = node.name
            output = forward_outputs[node_name]
            if self.cache_names is None or node_name in self.cache_names:
                if not self.allow_impure:
                    self._detect_impure(mapping[output], node_name)
                edges.append(CheckSumEdge(
                    self.checksums, node_name, self.get_storage(), self.return_tree,
                ).bind([output, key], node))
            else:
                edges.append(IdentityEdge().bind(output, node))

        return EdgesBag([key], outputs, edges, IdentityContext(), persistent_nodes=state.persistent_nodes)


class CheckSumIndex(CacheIndex):
    def get_hashes_tree(self, key):
        base = self.root / key_to_relative(key.digest, self.levels)
        data = base / DATA_FOLDER
        with self.locker.read(key.digest, base):
            assert data.exists(), data

            relative = {}
            for file in data.rglob('*'):
                if not file.is_dir():
                    with open(file, 'r') as fd:
                        relative[str(file.relative_to(data))] = fd.read()

            return relative


class CheckSumEdge(StaticGraph, StaticHash):
    def __init__(self, checksums: dict, base_key: str, storage, return_tree: bool):
        super().__init__(arity=2)
        self.base_key = base_key
        self.return_tree = return_tree
        self.cache = storage
        self._checksums = checksums

    def _make_hash(self, inputs):
        if self.return_tree:
            raise ValueError('In `return tree` mode graph hashing is not supported')
        return inputs[0]

    def evaluate(self):
        output = yield Command.ParentHash, 0
        key = self.cache.prepare(output)
        value, exists = self.cache.read(key, error=False)

        # populate the cache
        if not exists:
            value = yield Command.ParentValue, 0
            self.cache.write(key, value, error=False)

        # check consistency
        identifier = yield Command.ParentValue, 1
        tree = self._checksum(identifier, key)
        if self.return_tree:
            return tree
        return value

    def _checksum(self, identifier, key):
        if self.base_key not in self._checksums or identifier not in self._checksums[self.base_key]:
            # TODO: warn
            expected = None
        else:
            expected = self._checksums[self.base_key][identifier]

        for level in self.cache.levels:
            for location in level.locations:
                tree = location.get_hashes_tree(key)
                if tree is not None:
                    if expected is not None and tree != expected:
                        raise ValueError(
                            f'Checksum failed for {self.base_key}/{identifier}. '
                            f'Actual: {tree}, expected: {expected}'
                        )

                    return tree

        raise ValueError("Couldn't find the cached entry")
