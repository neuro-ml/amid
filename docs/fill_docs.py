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


records = []
root = Path(__file__).resolve().parent
with open(root / 'datasets-api.md', 'w') as file:
    file.write('# Datasets API\n\n')
    for name, (cls, module, description) in tqdm(list(gather_datasets().items())):
        file.write(f'::: {module}.{name}\n\n')
        entry = {'name': name, 'entries': len(cls().ids)}
        entry.update({k: v for k, v in description._asdict().items() if v is not None})
        link = entry.get('link')
        if link is not None:
            entry['link'] = f'<a href="{link}">Source</a>'

        records.append({k: stringify(v) for k, v in entry.items()})

table = pd.DataFrame.from_records(records)
table.columns = [x.replace('_', ' ').capitalize() for x in table.columns]
with open(root / 'datasets.md', 'w') as file:
    file.write('# Datasets\n\n')
    file.write(table.to_markdown(index=False))
