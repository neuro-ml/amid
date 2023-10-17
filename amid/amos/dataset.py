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
ERRORS = ['5514', '5437']  # these ids are damaged in the zip archives
# TODO: add MRI


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
    def ids(_id2split, _ids_unlabelled):
        labelled = sorted(_id2split)
        unlabelled = sorted(_ids_unlabelled)
        return labelled + unlabelled

    def image(i, _id2split, _base, _archive_name):
        """Corresponding 3D image."""
        if i in ERRORS:
            return None  # this image is damaged in the archive

        archive_name, archive_root = _archive_name
        if i in _id2split:
            archive_name = ARCHIVE_NAME_SEG
            archive_root = ARCHIVE_ROOT_NAME
            file = f'images{_id2split[i]}/amos_{i}.nii.gz'
        else:
            file = f'amos_{i}.nii.gz'

        with unpack(_base / archive_name, file, archive_root, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return np.asarray(nibabel.load(unpacked).dataobj)
            else:
                with open_nii_gz_file(unpacked) as image:
                    return np.asarray(image.dataobj)

    def affine(i, _id2split, _base, _archive_name):
        """The 4x4 matrix that gives the image's spatial orientation."""
        if i in ERRORS:
            return None  # this image is damaged in the archive
        archive_name, archive_root = _archive_name
        if i in _id2split:
            archive_name = ARCHIVE_NAME_SEG
            archive_root = ARCHIVE_ROOT_NAME
            file = f'images{_id2split[i]}/amos_{i}.nii.gz'
        else:
            file = f'amos_{i}.nii.gz'

        with unpack(_base / archive_name, file, archive_root, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return nibabel.load(unpacked).affine
            else:
                with open_nii_gz_file(unpacked) as image:
                    return image.affine

    def mask(i, _id2split, _base):
        if i in _id2split:
            file = f'labels{_id2split[i]}/amos_{i}.nii.gz'
        else:
            return

        try:
            with unpack(_base / ARCHIVE_NAME_SEG, file, ARCHIVE_ROOT_NAME, '.zip') as (unpacked, is_unpacked):
                if is_unpacked:
                    return np.asarray(nibabel.load(unpacked).dataobj)
                else:
                    with open_nii_gz_file(unpacked) as image:
                        return np.asarray(image.dataobj)
        except FileNotFoundError:
            return

    def image_modality(i):
        """Returns image modality, `CT` or `MRI`."""
        if 500 < int(i) <= 600:
            return 'MRI'
        return 'CT'

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

    def _ids_unlabelled(_base):
        ids_unlabelled = []
        for archive in [
            'amos22_unlabeled_ct_5000_5399.zip',
            'amos22_unlabeled_ct_5400_5899.zip',
            'amos22_unlabeled_ct_5900_6199.zip',
            'amos22_unlabeled_ct_6200_6899.zip',
        ]:
            with ZipFile(_base / archive) as zf:
                for x in zf.namelist():
                    if x.endswith('.nii.gz'):
                        file = x.split('/')[-1]
                        id_ = file.split('.')[0].split('_')[-1]
                        ids_unlabelled.append(id_)
        return ids_unlabelled

    @lru_cache(None)
    def _meta(_base):
        files = [
            'labeled_data_meta_0000_0599.csv',
            'unlabeled_data_meta_5400_5899.csv',
            'unlabeled_data_meta_5000_5399.csv',
            'unlabeled_data_meta_5900_6199.csv',
        ]

        dfs = []
        for file in files:
            with unpack(_base, file) as (unpacked, _):
                dfs.append(pd.read_csv(unpacked))
        return pd.concat(dfs)

    def _archive_name(i):
        if 5000 <= int(i) < 5400:
            return 'amos22_unlabeled_ct_5000_5399.zip', 'amos_unlabeled_ct_5000_5399'
        elif 5400 <= int(i) < 5900:
            return 'amos22_unlabeled_ct_5400_5899.zip', 'amos_unlabeled_ct_5400_5899'
        elif 5900 <= int(i) < 6200:
            return 'amos22_unlabeled_ct_5900_6199.zip', 'amos22_unlabeled_ct_5900_6199'
        else:
            return 'amos22_unlabeled_ct_6200_6899.zip', 'amos22_unlabeled_6200_6899'


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
    raw_data_size='23G',  # TODO: update size with unlabelled
    prep_data_size='89,5G',
    task='Supervised multi-modality abdominal multi-organ segmentation',
    normalizers=[SpacingFromAffine()],
)
