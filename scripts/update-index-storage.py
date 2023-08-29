import argparse

from deli import load
from tarn import HashKeyStorage
from tqdm.auto import tqdm

from amid.internals import get_repo


parser = argparse.ArgumentParser()
parser.add_argument('root')
root = parser.parse_args().root

repo = get_repo()
src = repo.storage
dst = HashKeyStorage(root, algorithm='blake2b')

for file in tqdm(repo.path.rglob('*.hash'), 'Gathering'):
    key = load(file, hint='.txt').split(':')[1]
    dst.write(src.read(lambda x: x.read(), key))
