import os
import re
from pathlib import Path

import deli
import pandas as pd
from tqdm import tqdm

from amid.internals.registry import gather_datasets, prepare_for_table


file = Path(__file__).resolve().parent.parent / 'README.md'
with open(file, 'r') as fd:
    content = fd.read()

start = re.search(r'# Available datasets', content).end()
stop = re.search(r'Check out \[our docs\]', content).start()
raw_data = deli.load(os.environ.get('RAW_DATA'))
cache = deli.load('cache.json')

records = []
for name, (cls, module, description) in tqdm(list(gather_datasets().items())):  # noqa
    if name in cache:
        count = cache[name]
    else:
        count = len(cls(root=raw_data[name]).ids)
        cache[name] = count
        deli.save(cache, 'cache.json')
    records.append(prepare_for_table(name, count, module, description, 'latest'))

table = pd.DataFrame.from_records(records).fillna('')
table.columns = [x.replace('_', ' ').capitalize() for x in table.columns]
table = table[['Name', 'Entries', 'Body region', 'Modality']].to_markdown(index=False)
content = f'{content[:start]}\n\n{table}\n\n{content[stop:]}'

with open(file, 'w') as fd:
    fd.write(content)
