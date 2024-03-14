from functools import partial

import nibabel
import numpy as np
from connectome import Function

from ..utils import open_nii_gz_file, unpack
from .const import ANATOMICAL_STRUCTURES, LABELS


ARCHIVE_ROOT = 'Totalsegmentator_dataset'


def label_loader(name, i, _meta):
    return _meta[_meta['image_id'] == i][name].item()


def mask_loader(name, i, _base):
    file = f'{i}/segmentations/{name}.nii.gz'

    with unpack(_base, file, ARCHIVE_ROOT, '.zip') as (unpacked, is_unpacked):
        if is_unpacked:
            return np.asarray(nibabel.load(unpacked).dataobj)
        else:
            with open_nii_gz_file(unpacked) as image:
                return np.asarray(image.dataobj)


def add_labels(scope):
    for label in LABELS:
        scope[label] = Function(partial(label_loader, label), 'id', '_meta')


def add_masks(scope):
    for anatomical_structure in ANATOMICAL_STRUCTURES:
        scope[anatomical_structure] = Function(partial(mask_loader, anatomical_structure), 'id', '_base')
