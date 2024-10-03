# Contribution Guidelines

## Preparing the environment

1\. First, set up a cache storage. Create the file `~/.config/amid/.bev.yml` with the following content:

```yaml
main:
  storage: /path/to/storage
  cache: /path/to/cache
```

where `/path/to/storage` and `/path/to/cache` are some paths in your filesystem.

2\. Run

```shell
amid init
```

The full command could look something like this:

```shell
mkdir -p ~/.config/amid
cat >~/.config/amid/.bev.yml <<EOL
main:
  storage: /mount/data/storage
  cache: /mount/data/cache
EOL
amid init
```

## Adding a new dataset

We will be using [LiTS](https://github.com/neuro-ml/amid/blob/master/amid/lits.py) as an example.

1\. Download the raw data to a separate folder in your filesystem

2\. (Optionally) create a new branch for the dataset:

```shell
git checkout lits
```

3\. Create a class that loads the raw data. [LiTS](https://github.com/neuro-ml/amid/blob/master/amid/lits.py) is a good
example. Note how each field is implemented as a separate function.

There are no strict rules regarding the dataset fields,
but try to keep the output "as raw as possible",
i.e., do not apply heavy processing that modifies the data irreversibly.

**Rule of thumb:**

> The dataset should be written in such a way, that making a submission to a contest would work out of the box.

!!! note
    In case of DICOM files, make sure to transpose the first 2 image axes. 
    This way, the image axes will be consistent with the potential contour coordinates.

!!! tip 
    If some value is missing for a given id, it is preferable to return `None` instead of raising an exception.

!!! tip
    The dataset must have a docstring which describes it and provides a link to the original data.

!!! tip
    If the raw data contains a table with metadata, it is preferable to split the metadata columns into separate fields.

4\. Register the dataset like so:

```python
from amid.internals import register

@register(
    ...,
)
class LiTS(Dataset):
    ...
```

where `...` stands for the following arguments:

- `modality` — the images' modality/modalities, e.g., CT, MRI
- `body_region` — the anatomical regions present in the dataset, e.g., Head, Thorax, Abdomen
- `license` — the dataset's license, if any
- `link` — the link to the original data
- `raw_data_size` — the total size, required for the raw data, e.g., 10G, 500M
- `task` — the dataset's downstream task if any.
    E.g., Supervised Learning, Domain Adaptation, Self-supervised Learning, Tumor Segmentation, etc.

5\. Make sure all the methods are working as expected:

```python
from amid.lits import LiTS

dataset = LiTS(root="/datasets/LiTS")

print(len(dataset.ids))

id_ = dataset.ids[0]
print(dataset.image(id_).shape)
```

6\. Check the codestyle using the `lint.sh` script in the repository's root and make changes if flake8 is not happy:

```shell
pip install -r lint-requirements.txt # only for the first time
./lint.sh
```