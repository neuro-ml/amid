from pathlib import Path

import pandas as pd
from tqdm import tqdm

from amid.internals.registry import gather_datasets, prepare_for_table

records = []
root = Path(__file__).resolve().parent
with open(root / 'datasets-api.md', 'w') as file:
    file.write('# Datasets API\n\n')
    for name, (cls, module, description) in tqdm(list(gather_datasets().items())):
        file.write(f'::: {module}.{name}\n\n')
        records.append(prepare_for_table(name, cls, description))

table = pd.DataFrame.from_records(records).fillna('')
table.columns = [x.replace('_', ' ').capitalize() for x in table.columns]
with open(root / 'datasets.md', 'w') as file:
    file.write('# Datasets\n\n')
    file.write(table.to_markdown(index=False))
