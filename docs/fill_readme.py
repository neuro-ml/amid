import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from amid.internals.registry import gather_datasets


def stringify(x):
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return ', '.join(x)
    return x


file = Path(__file__).resolve().parent.parent / 'README.md'
with open(file, 'r') as fd:
    content = fd.read()

start = re.search(r'# Available datasets', content).end()
stop = re.search(r'Check out \[our docs\]', content).start()

records = []
for name, (cls, module, description) in tqdm(list(gather_datasets().items())):  # noqa
    entry = {'name': name, 'entries': len(cls().ids)}
    entry.update({k: v for k, v in description._asdict().items() if v is not None})
    link = entry.get('link')
    if link is not None:
        entry['name'] = f'<a href="{link}">{name}</a>'

    records.append({k: stringify(v) for k, v in entry.items()})

table = pd.DataFrame.from_records(records)
table.columns = [x.replace('_', ' ').capitalize() for x in table.columns]
table = table[['Name', 'Entries', 'Body region', 'Modality']].to_markdown(index=False)
content = f'{content[:start]}\n\n{table}\n\n{content[stop:]}'

with open(file, 'w') as fd:
    fd.write(content)
