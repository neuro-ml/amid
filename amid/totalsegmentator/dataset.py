from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import checksum, register
from .utils import add_labels, add_masks, open_nii_gz_file, unpack


@register(
    body_region=('Head', 'Thorax', 'Abdomen', 'Pelvis', 'Legs'),
    license='CC BY 4.0',
    link='https://zenodo.org/record/6802614#.Y6M2MxXP1D8',
    modality='CT',
    raw_data_size='35G',
    prep_data_size='35G',
    task='Supervised anatomical structures segmentation',
)
@checksum('totalsegmentator')
class Totalsegmentator(Source):
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

    _root: str = None

    add_masks(locals())
    add_labels(locals())

    def _base(_root: Silent):
        _root = Path(_root)
        if _root.is_dir():
            if list(_root.iterdir()) == ['Totalsegmentator_dataset']:
                return _root / 'Totalsegmentator_dataset'
        # it's a zip file
        return _root

    @meta
    def ids(_base):
        if _base.is_dir():
            namelist = [x.name for x in _base.glob('*')]
        else:
            with ZipFile(_base) as zf:
                namelist = [x.rstrip('/') for x in zf.namelist()]

        ids = []
        for f in namelist:
            if len(f.split('/')) == 2 and f.split('/')[-1] != 'meta.csv':
                ids.append(f.split('/')[-1])

        return sorted(ids)

    def image(i, _base):
        file = f'{i}/ct.nii.gz'

        with unpack(_base, file) as (unpacked, is_unpacked):
            if is_unpacked:
                return np.asarray(nibabel.load(unpacked).dataobj)
            else:
                with open_nii_gz_file(unpacked) as image:
                    return np.asarray(image.dataobj)

    def affine(i, _base):
        """The 4x4 matrix that gives the image's spatial orientation"""
        file = f'{i}/ct.nii.gz'

        with unpack(_base, file) as (unpacked, is_unpacked):
            if is_unpacked:
                return nibabel.load(unpacked).affine
            else:
                with open_nii_gz_file(unpacked) as image:
                    return image.affine

    @lru_cache(None)
    def _meta(_base):
        file = 'meta.csv'

        with unpack(_base, file) as (unpacked, _):
            return pd.read_csv(unpacked, sep=';')
