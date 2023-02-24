import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from amid.internals.registry import gather_datasets, prepare_for_table

version = os.environ.get('VERSION')
if not version:
    raise RuntimeError('Please define the "VERSION" env variable')

records = []
root = Path(__file__).resolve().parent
with open(root / 'datasets-api.md', 'w') as file:
    file.write('# Datasets API\n\n')
    for name, (cls, module, description) in tqdm(list(gather_datasets().items())):
        file.write(f'::: {module}.{name}\n\n')
        records.append(prepare_for_table(name, cls, module, description, version))

table = pd.DataFrame.from_records(records).fillna('')
table.columns = [x.replace('_', ' ').capitalize() for x in table.columns]
with open(root / 'datasets.md', 'w') as file:
    file.write('# Datasets\n\n')
    file.write(table.to_markdown(index=False))
