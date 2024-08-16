import os
from pathlib import Path

import deli
import pandas as pd
from tqdm import tqdm

from amid.__version__ import __version__ as version
from amid.internals.registry import gather_datasets, prepare_for_table


# version = os.environ.get('VERSION')
# if not version:
#     raise RuntimeError('Please define the "VERSION" env variable')
raw_data = deli.load(os.environ.get('RAW_DATA'))
cache = deli.load('cache.json')

records = []
root = Path(__file__).resolve().parent
with open(root / 'datasets-api.md', 'w') as file:
    file.write('# Datasets API\n\n')
    for name, (cls, module, description) in tqdm(list(gather_datasets().items())):
        file.write(f'::: {module}.{name}\n\n')
        if name in cache:
            count = cache[name]
        else:
            count = len(cls(root=raw_data[name]).ids)
            cache[name] = count
            deli.save(cache, 'cache.json')

        records.append(prepare_for_table(name, count, module, description, version))

table = pd.DataFrame.from_records(records).fillna('')
table.columns = [x.replace('_', ' ').capitalize() for x in table.columns]
with open(root / 'datasets.md', 'w') as file:
    file.write('# Datasets\n\n')
    file.write(table.to_markdown(index=False))
