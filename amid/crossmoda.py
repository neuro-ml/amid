import contextlib
import gzip
import zipfile
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import nibabel as nb
import numpy as np
import pandas as pd

from .internals import Dataset, licenses, register


@register(
    body_region='Head',
    license=licenses.CC_BYNCSA_40,
    link='https://zenodo.org/record/6504722#.YsgwnNJByV4',
    modality=('MRI T1c', 'MRI T2hr'),
    prep_data_size='8,96G',
    raw_data_size='17G',
    task=('Segmentation', 'Classification', 'Domain Adaptation'),
)
class CrossMoDA(Dataset):
    """
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download links:
    2021 & 2022: https://zenodo.org/record/6504722#.YsgwnNJByV4

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = CrossMoDA(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 484
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 214)

    References
    ----------
    """

    @property
    def ids(self):
        result = set()
        for archive in self.root.glob('*.zip'):
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue

                    file = Path(zipinfo.filename)
                    assert file.stem not in result, file.stem

                    if 'Label' not in file.stem and file.suffix == '.gz':
                        result.add('_'.join(file.stem.split('_')[:-1]))
                    else:
                        continue

        return sorted(result)

    @property
    def train_source_df(self):
        return pd.read_csv(self.root / 'infos_source_training.csv', index_col='crossmoda_name')

    def _file(self, i):
        for archive in self.root.glob('*.zip'):
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if i == '_'.join(Path(zipinfo.filename).stem.split('_')[:-1]) and 'Label' not in zipinfo.filename:
                        return zipfile.Path(archive, zipinfo.filename)

        raise ValueError(f'Id "{i}" not found')

    def image(self, i) -> Union[np.ndarray, None]:
        with open_nii_gz_file(self._file(i)) as nii_image:
            return np.asarray(nii_image.dataobj)

    def spacing(self, i):
        """Returns pixel spacing along axes (x, y, z)"""
        with open_nii_gz_file(self._file(i)) as nii_image:
            return tuple(nii_image.header['pixdim'][1:4])

    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation"""
        with open_nii_gz_file(self._file(i)) as nii_image:
            return nii_image.affine

    def split(self, i) -> str:
        """The split in which this entry is contained: training_source, training_target, validation"""
        file = self._file(i)
        idx = int(file.name.split('_')[2])
        dataset = file.name.split('_')[1]

        if dataset == 'ldn':
            if 1 <= idx < 106:
                return 'training_source'
            elif 106 <= idx < 211:
                return 'training_target'
            elif 211 <= idx < 243:
                return 'validation'

        elif dataset == 'etz':
            if 0 <= idx < 105:
                return 'training_source'
            elif 105 <= idx < 210:
                return 'training_target'
            elif 210 <= idx < 242:
                return 'validation'

        raise ValueError(f'Cannot find split for the file: {file}')

    def year(self, i) -> int:
        """The year in which this entry was published: 2021 or 2022"""
        return int(self._file(i).name[9:13])

    def masks(self, i):
        """Combined mask of schwannoma and cochlea (1 and 2 respectively)"""
        file = self._file(i)
        if 'T2' not in file.name:
            with open_nii_gz_file(file.parent / file.name.replace('ceT1', 'Label')) as nii_image:
                return nii_image.get_fdata().astype(np.uint8)

    def koos_grade(self, i):
        """VS Tumour characteristic according to Koos grading scale: [1..4] or (-1 - post operative)"""
        if self.split(i) == 'training_source':
            grade = self.train_source_df.loc[i, 'koos']
            return -1 if (grade == 'post-operative-london') else int(grade)


# TODO: sync with amid.utils
@contextlib.contextmanager
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
