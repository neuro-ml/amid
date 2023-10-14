from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd
from connectome import Source, Transform, meta
from connectome.interface.nodes import Silent

from ..internals import licenses, normalize
from ..utils import open_nii_gz_file, unpack
from .utils import label


ARCHIVE_NAME_SEG = 'amos22.zip'
ARCHIVE_ROOT_NAME = 'amos22'


class AMOSBase(Source):
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

    def _base(_root: Silent):
        return Path(_root)

    @meta
    def ids(_id2split):
        return sorted(_id2split)

    def image(i, _id2split, _base, _archive_name):
        if i in _id2split:
            archive_name = ARCHIVE_NAME_SEG
            archive_root = ARCHIVE_ROOT_NAME
            file = f'images{_id2split[i]}/amos_{i}.nii.gz'
        else: 
            archive_name = _archive_name
            archive_root = '.'
            file = f'amos_{i}.nii.gz'
        

        with unpack(_base / archive_name, file, archive_root, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return np.asarray(nibabel.load(unpacked).dataobj)
            else:
                with open_nii_gz_file(unpacked) as image:
                    return np.asarray(image.dataobj)

    def affine(i, _id2split, _base, _archive_name):
        """The 4x4 matrix that gives the image's spatial orientation"""
        if i in _id2split:
            archive_name = ARCHIVE_NAME_SEG
            archive_root = ARCHIVE_ROOT_NAME
            file = f'images{_id2split[i]}/amos_{i}.nii.gz'
        else: 
            archive_name = _archive_name
            archive_root = '.'
            file = f'amos_{i}.nii.gz'

        with unpack(_base / archive_name, file, archive_root, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return nibabel.load(unpacked).affine
            else:
                with open_nii_gz_file(unpacked) as image:
                    return image.affine

    def mask(i, _id2split, _base):
        file = f'labels{_id2split[i]}/amos_{i}.nii.gz'

        try:
            with unpack(_base / ARCHIVE_NAME_SEG, file, ARCHIVE_ROOT_NAME, '.zip') as (unpacked, is_unpacked):
                if is_unpacked:
                    return np.asarray(nibabel.load(unpacked).dataobj)
                else:
                    with open_nii_gz_file(unpacked) as image:
                        return np.asarray(image.dataobj)
        except FileNotFoundError:
            return None

    # labels

    birth_date = label("Patient's Birth Date")
    sex = label("Patient's Sex")
    age = label("Patient's Age")
    manufacturer_model = label("Manufacturer's Model Name")
    manufacturer = label('Manufacturer')
    acquisition_date = label('Acquisition Date')
    site = label('Site')

    @lru_cache(None)
    def _id2split(_base):
        id2split = {}

        with ZipFile(_base / ARCHIVE_NAME_SEG) as zf:
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
        
    def _archive_name(i):
        if  5000 <= int(i) < 5400:
            return 'amos22_unlabeled_ct_5000_5399.zip'
        elif 5400 <= int(i) < 5900:
            return 'amos22_unlabeled_ct_5400_5899.zip'
        elif 5900 <= int(i) < 6200:
            return 'amos22_unlabeled_ct_5900_6199.zip'
        else:
            return 'amos22_unlabeled_ct_6200_6899.zip'
        

class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nibabel.affines.voxel_sizes(affine)


AMOS = normalize(
    AMOSBase,
    'AMOS',
    'amos',
    body_region='Abdomen',
    license=licenses.CC_BY_40,
    link='https://zenodo.org/record/7262581',
    modality=('CT', 'MRI'),
    raw_data_size='23G',
    prep_data_size='89,5G',
    task='Supervised multi-modality abdominal multi-organ segmentation',
    normalizers=[SpacingFromAffine()],
)
