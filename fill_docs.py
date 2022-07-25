from pathlib import Path
import pandas as pd

from amid.internals.registry import gather_datasets, Description

records = []
root = Path(__file__).resolve().parent
with open(root / 'datasets-api.md', 'w') as file:
    file.write('# Datasets API\n\n')
    for name, (cls, module, kwargs) in gather_datasets().items():
        file.write(f'::: {module}.{name}\n\n')
        records.append(Description(name=name, entries=len(cls().ids), **kwargs)._asdict())

table = pd.DataFrame.from_records(records)
table.columns = [x.capitalize() for x in table.columns]
with open(root / 'datasets.md', 'w') as file:
    file.write('# Datasets\n\n')
    file.write(table.to_markdown(index=False))
