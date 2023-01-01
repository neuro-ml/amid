import nibabel
import numpy as np

from ..utils import open_nii_gz_file, unpack
from .const import ANATOMICAL_STRUCTURES, LABELS


ARCHIVE_ROOT = 'Totalsegmentator_dataset'


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

            with unpack(_base, file, ARCHIVE_ROOT, '.zip') as (unpacked, is_unpacked):
                if is_unpacked:
                    return np.asarray(nibabel.load(unpacked).dataobj)
                else:
                    with open_nii_gz_file(unpacked) as image:
                        return np.asarray(image.dataobj)

        return loader

    for anatomical_structure in ANATOMICAL_STRUCTURES:
        scope[anatomical_structure] = make_loader(anatomical_structure)
