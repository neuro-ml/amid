from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import checksum, licenses, register
from ..utils import open_nii_gz_file, unpack
from .utils import add_labels


ARCHIVE_NAME = 'amos22.zip'
ARCHIVE_ROOT_NAME = 'amos22'


@register(
    body_region='Abdomen',
    license=licenses.CC_BY_40,
    link='https://zenodo.org/record/7262581',
    modality=('CT', 'MRI'),
    raw_data_size='23G',
    prep_data_size='89,5G',
    task='Supervised multi-modality abdominal multi-organ segmentation',
)
@checksum('amos')
class AMOS(Source):
    """
    AMOS provides 500 CT and 100 MRI scans collected from multi-center, multi-vendor, multi-modality, multi-phase,
    multi-disease patients, each with voxel-level annotations of 15 abdominal organs, providing challenging examples
    and test-bed for studying robust segmentation algorithms under diverse targets and scenarios. [1]

    Parameters
    ----------
    root : str, Path, optional
        Absolute path to the root containing the downloaded archive and meta.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download link: https://zenodo.org/record/7262581/files/amos22.zip

    Examples
    --------
    >>> # Download the archive and meta to any folder and pass the path to the constructor:
    >>> ds = AMOS(root='/path/to/the/downloaded/files')
    >>> print(len(ds.ids))
    # 961
    >>> print(ds.image(ds.ids[0]).shape)
    # (768, 768, 90)
    >>> print(ds.mask(ds.ids[26]).shape)
    # (512, 512, 124)

    References
    ----------
    .. [1] JI YUANFENG. (2022). Amos: A large-scale abdominal multi-organ benchmark for
    versatile medical image segmentation [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7262581
    """

    _root: str = None

    add_labels(locals())

    def _base(_root: Silent):
        return Path(_root)

    @meta
    def ids(_id2split):
        return sorted(_id2split)

    def image(i, _id2split, _base):
        file = f'images{_id2split[i]}/amos_{i}.nii.gz'

        with unpack(_base / ARCHIVE_NAME, file, ARCHIVE_ROOT_NAME, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return np.asarray(nibabel.load(unpacked).dataobj)
            else:
                with open_nii_gz_file(unpacked) as image:
                    return np.asarray(image.dataobj)

    def affine(i, _id2split, _base):
        """The 4x4 matrix that gives the image's spatial orientation"""
        file = f'images{_id2split[i]}/amos_{i}.nii.gz'

        with unpack(_base / ARCHIVE_NAME, file, ARCHIVE_ROOT_NAME, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return nibabel.load(unpacked).affine
            else:
                with open_nii_gz_file(unpacked) as image:
                    return image.affine

    def mask(i, _id2split, _base):
        file = f'labels{_id2split[i]}/amos_{i}.nii.gz'

        try:
            with unpack(_base / ARCHIVE_NAME, file, ARCHIVE_ROOT_NAME, '.zip') as (unpacked, is_unpacked):
                if is_unpacked:
                    return np.asarray(nibabel.load(unpacked).dataobj)
                else:
                    with open_nii_gz_file(unpacked) as image:
                        return np.asarray(image.dataobj)
        except FileNotFoundError:
            return None

    @lru_cache(None)
    def _id2split(_base):
        id2split = {}

        with ZipFile(_base / ARCHIVE_NAME) as zf:
            for x in zf.namelist():
                if (len(x.strip('/').split('/')) == 3) and x.endswith('.nii.gz'):
                    file, split = x.split('/')[-1], x.split('/')[-2][-2:]
                    id_ = file.split('.')[0].split('_')[-1]

                    id2split[id_] = split

        return id2split

    @lru_cache(None)
    def _meta(_base):
        file = 'labeled_data_meta_0000_0599.csv'

        with unpack(_base, file) as (unpacked, _):
            return pd.read_csv(unpacked)
