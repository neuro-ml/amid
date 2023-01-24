import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from amid.internals.registry import gather_datasets, prepare_for_table


file = Path(__file__).resolve().parent.parent / 'README.md'
with open(file, 'r') as fd:
    content = fd.read()

start = re.search(r'# Available datasets', content).end()
stop = re.search(r'Check out \[our docs\]', content).start()

records = []
for name, (cls, module, description) in tqdm(list(gather_datasets().items())):  # noqa
    records.append(prepare_for_table(name, cls, description))

table = pd.DataFrame.from_records(records).fillna('')
table.columns = [x.replace('_', ' ').capitalize() for x in table.columns]
table = table[['Name', 'Entries', 'Body region', 'Modality']].to_markdown(index=False)
content = f'{content[:start]}\n\n{table}\n\n{content[stop:]}'

with open(file, 'w') as fd:
    fd.write(content)
