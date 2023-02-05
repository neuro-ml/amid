import tomllib
from pathlib import Path


root = Path(__file__).resolve().parent.parent
with open(root / 'requirements.txt', 'r') as file:
    first = set(filter(None, (x.strip() for x in file.readlines())))
with open(root / 'pyproject.toml', 'rb') as file:
    second = set(tomllib.load(file)['project']['dependencies'])

if first != second:
    raise ValueError(f'The reqs diverge: {first ^ second}')
