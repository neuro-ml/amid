import nibabel
import numpy as np

from ..internals.dataset import register_field
from ..utils import open_nii_gz_file, unpack
from .const import ANATOMICAL_STRUCTURES, LABELS


ARCHIVE_ROOT = 'Totalsegmentator_dataset'


def label_loader(name):
    def loader(self, i):
        return self._meta[self._meta['image_id'] == i][name].item()

    register_field('Totalsegmentator', name, loader)
    return loader


def mask_loader(name):
    def loader(self, i):
        file = f'{i}/segmentations/{name}.nii.gz'

        with unpack(self.root, file, ARCHIVE_ROOT, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return np.asarray(nibabel.load(unpacked).dataobj)
            else:
                with open_nii_gz_file(unpacked) as image:
                    return np.asarray(image.dataobj)

    register_field('Totalsegmentator', name, loader)
    return loader


def add_labels(scope):
    for label in LABELS:
        scope[label] = label_loader(label)


def add_masks(scope):
    for anatomical_structure in ANATOMICAL_STRUCTURES:
        scope[anatomical_structure] = mask_loader(anatomical_structure)
