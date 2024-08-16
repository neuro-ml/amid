from functools import cached_property
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd
from jboc import composed

from ..internals import Dataset, field, licenses, register
from ..utils import open_nii_gz_file, unpack


ARCHIVE_NAME_SEG = 'amos22.zip'
ARCHIVE_ROOT_NAME = 'amos22'
ERRORS = ['5514', '5437']  # these ids are damaged in the zip archives


# TODO: add MRI


@register(
    body_region='Abdomen',
    license=licenses.CC_BY_40,
    link='https://zenodo.org/record/7262581',
    modality=('CT', 'MRI'),
    raw_data_size='23G',  # TODO: update size with unlabelled
    prep_data_size='89,5G',
    task='Supervised multi-modality abdominal multi-organ segmentation',
)
class AMOS(Dataset):
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

    @property
    def ids(self):
        ids = list(self._id2split)

        for archive in [
            'amos22_unlabeled_ct_5000_5399.zip',
            'amos22_unlabeled_ct_5400_5899.zip',
            'amos22_unlabeled_ct_5900_6199.zip',
            'amos22_unlabeled_ct_6200_6899.zip',
        ]:
            file = self.root / archive
            if not file.exists():
                continue

            with ZipFile(file) as zf:
                for x in zf.namelist():
                    if x.endswith('.nii.gz'):
                        file = x.split('/')[-1]

                        ids.append(file.split('.')[0].split('_')[-1])

        return sorted(ids)

    @field
    def image(self, i):
        """Corresponding 3D image."""
        if i in ERRORS:
            return None  # this image is damaged in the archive

        archive_name, archive_root, file = self._archive_name(i)
        with unpack(self.root / archive_name, file, archive_root, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return np.asarray(nibabel.load(unpacked).dataobj)
            else:
                with open_nii_gz_file(unpacked) as image:
                    return np.asarray(image.dataobj)

    @field
    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation."""
        if i in ERRORS:
            return None  # this image is damaged in the archive

        archive_name, archive_root, file = self._archive_name(i)
        with unpack(self.root / archive_name, file, archive_root, '.zip') as (unpacked, is_unpacked):
            if is_unpacked:
                return nibabel.load(unpacked).affine
            else:
                with open_nii_gz_file(unpacked) as image:
                    return image.affine

    @field
    def mask(self, i):
        if i not in self._id2split:
            return

        file = f'labels{self._id2split[i]}/amos_{i}.nii.gz'
        try:
            with unpack(self.root / ARCHIVE_NAME_SEG, file, ARCHIVE_ROOT_NAME, '.zip') as (unpacked, is_unpacked):
                if is_unpacked:
                    return np.asarray(nibabel.load(unpacked).dataobj)
                else:
                    with open_nii_gz_file(unpacked) as image:
                        return np.asarray(image.dataobj)
        except FileNotFoundError:
            return

    @field
    def image_modality(self, i):
        """Returns image modality, `CT` or `MRI`."""
        if 500 < int(i) <= 600:
            return 'MRI'
        return 'CT'

    # labels
    @field
    def birth_date(self, i):
        return self._label(i, "Patient's Birth Date")

    @field
    def sex(self, i):
        return self._label(i, "Patient's Sex")

    @field
    def age(self, i):
        return self._label(i, "Patient's Age")

    @field
    def manufacturer_model(self, i):
        return self._label(i, "Manufacturer's Model Name")

    @field
    def manufacturer(self, i):
        return self._label(i, 'Manufacturer')

    @field
    def acquisition_date(self, i):
        return self._label(i, 'Acquisition Date')

    @field
    def site(self, i):
        return self._label(i, 'Site')

    @cached_property
    @composed(dict)
    def _id2split(self):
        with ZipFile(self.root / ARCHIVE_NAME_SEG) as zf:
            for x in zf.namelist():
                if (len(x.strip('/').split('/')) == 3) and x.endswith('.nii.gz'):
                    file, split = x.split('/')[-1], x.split('/')[-2][-2:]
                    id_ = file.split('.')[0].split('_')[-1]

                    yield id_, split

    @cached_property
    def _meta(self):
        files = [
            'labeled_data_meta_0000_0599.csv',
            'unlabeled_data_meta_5400_5899.csv',
            'unlabeled_data_meta_5000_5399.csv',
            'unlabeled_data_meta_5900_6199.csv',
        ]

        dfs = []
        for file in files:
            with unpack(self.root, file) as (unpacked, _):
                dfs.append(pd.read_csv(unpacked))
        return pd.concat(dfs)

    def _archive_name(self, i):
        if i in self._id2split:
            return ARCHIVE_NAME_SEG, ARCHIVE_ROOT_NAME, f'images{self._id2split[i]}/amos_{i}.nii.gz'

        i = int(i)
        file = f'amos_{i}.nii.gz'
        if 5000 <= i < 5400:
            return 'amos22_unlabeled_ct_5000_5399.zip', 'amos_unlabeled_ct_5000_5399', file
        elif 5400 <= i < 5900:
            return 'amos22_unlabeled_ct_5400_5899.zip', 'amos_unlabeled_ct_5400_5899', file
        elif 5900 <= i < 6200:
            return 'amos22_unlabeled_ct_5900_6199.zip', 'amos22_unlabeled_ct_5900_6199', file
        else:
            return 'amos22_unlabeled_ct_6200_6899.zip', 'amos22_unlabeled_6200_6899', file

    def _label(self, i, column):
        # ambiguous data in meta
        if int(i) in [500, 600]:
            return None
        elif int(i) not in self._meta['amos_id']:
            return None

        return self._meta[self._meta['amos_id'] == int(i)][column].item()
