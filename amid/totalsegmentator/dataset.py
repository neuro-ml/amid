import gzip
from contextlib import suppress
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd

from ..internals import Dataset, field, licenses, register
from ..utils import PathOrStr, open_nii_gz_file, unpack
from .utils import ARCHIVE_ROOT, add_labels, add_masks


@register(
    body_region=('Head', 'Thorax', 'Abdomen', 'Pelvis', 'Legs'),
    license=licenses.CC_BY_40,
    link='https://zenodo.org/record/6802614#.Y6M2MxXP1D8',
    modality='CT',
    raw_data_size='35G',
    prep_data_size='35G',
    task='Supervised anatomical structures segmentation',
)
class Totalsegmentator(Dataset):
    """
    In 1204 CT images we segmented 104 anatomical structures (27 organs, 59 bones, 10 muscles, 8 vessels)
    covering a majority of relevant classes for most use cases.

    The CT images were randomly sampled from clinical routine, thus representing a real world dataset which
    generalizes to clinical application.

    The dataset contains a wide range of different pathologies, scanners, sequences and institutions. [1]

    Parameters
    ----------
    root : str, Path, optional
        absolute path to the downloaded archive.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download link: https://zenodo.org/record/6802614/files/Totalsegmentator_dataset.zip

    Examples
    --------
    >>> # Download the archive to any folder and pass the path to the constructor:
    >>> ds = Totalsegmentator(root='/path/to/the/downloaded/archive')
    >>> print(len(ds.ids))
    # 1204
    >>> print(ds.image(ds.ids[0]).shape)
    # (294, 192, 179)
    >>> print(ds.aorta(ds.ids[25]).shape)
    # (320, 320, 145)

    References
    ----------
    .. [1] Jakob Wasserthal (2022) Dataset with segmentations of 104 important anatomical structures in 1204 CT images.
    Available at: https://zenodo.org/record/6802614#.Y6M2MxXP1D8
    """

    add_masks(locals())
    add_labels(locals())

    def __init__(self, root: PathOrStr):
        root = Path(root)
        if root.is_dir():
            if root / ARCHIVE_ROOT in list(root.iterdir()):
                root = root / ARCHIVE_ROOT

        file = 'meta.csv'
        with unpack(root, file, ARCHIVE_ROOT, '.zip') as (unpacked, _):
            self._meta = pd.read_csv(unpacked, sep=';')

        super().__init__(root)

    @property
    def ids(self):
        if self.root.is_dir():
            return sorted({x.name for x in self.root.iterdir() if x.name != 'meta.csv'})
        else:
            with ZipFile(self.root) as zf:
                parsed_namelist = [x.strip('/').split('/') for x in zf.namelist()]
                return sorted({x[-1] for x in parsed_namelist if len(x) == 2 and x[-1] != 'meta.csv'})

    @field
    def image(self, i):
        file = f'{i}/ct.nii.gz'

        with suppress(gzip.BadGzipFile):
            with unpack(self.root, file, ARCHIVE_ROOT, '.zip') as (unpacked, is_unpacked):
                if is_unpacked:
                    return np.asarray(nibabel.load(unpacked).dataobj)
                else:
                    with open_nii_gz_file(unpacked) as image:
                        return np.asarray(image.dataobj)

    @field
    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation"""
        file = f'{i}/ct.nii.gz'

        with unpack(self.root, file, ARCHIVE_ROOT, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return nibabel.load(unpacked).affine
            else:
                with open_nii_gz_file(unpacked) as image:
                    return image.affine
