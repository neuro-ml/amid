Awesome Medical Imaging Datasets (AMID) - a curated list of medical imaging datasets with unified interfaces

## Getting started

Just import a dataset and start using it!

Note that for some datasets you must manually download the raw files first.

```python
from amid.verse import VerSe

ds = VerSe()
# get the available ids
print(len(ds.ids))
i = ds.ids[0]

# use the available methods:
#   load the image and vertebrae masks
x, y = ds.image(i), ds.masks(i)
print(ds.split(i), ds.patient(i))

# or get a namedTuple-like object:
entry = ds(i)
x, y = entry.image, entry.masks
print(entry.split, entry.patient)
```

## Install

Just get it from PyPi:

```shell
pip install amid
```

Or if you want to use version control features:

```shell
git clone https://github.com/neuro-ml/amid.git
cd amid && pip install -e .
```
