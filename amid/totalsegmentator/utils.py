import contextlib
import gzip
import zipfile
from pathlib import Path

import nibabel
import numpy as np

from .const import ANATOMICAL_STRUCTURES, LABELS


def add_labels(scope):
    def make_loader(label):
        def loader(i, _meta):
            return _meta[_meta['image_id'] == i][label].item()

        return loader

    for label in LABELS:
        scope[label] = make_loader(label)


def add_masks(scope):
    def make_loader(anatomical_structure):
        def loader(i, _base):
            file = f'{i}/segmentations/{anatomical_structure}.nii.gz'

            with unpack(_base, file) as (unpacked, is_unpacked):
                if is_unpacked:
                    return np.asarray(nibabel.load(unpacked).dataobj)
                else:
                    with open_nii_gz_file(unpacked) as image:
                        return np.asarray(image.dataobj)

        return loader

    for anatomical_structure in ANATOMICAL_STRUCTURES:
        scope[anatomical_structure] = make_loader(anatomical_structure)


@contextlib.contextmanager
def unpack(root: str, relative: str):
    unpacked = Path(root) / relative

    if unpacked.exists():
        yield unpacked, True
    else:
        with zipfile.Path(root, str(Path('Totalsegmentator_dataset', relative))).open('rb') as unpacked:
            yield unpacked, False


@contextlib.contextmanager
def open_nii_gz_file(unpacked):
    with gzip.GzipFile(fileobj=unpacked) as nii:
        nii = nibabel.FileHolder(fileobj=nii)
        yield nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
