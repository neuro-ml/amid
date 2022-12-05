import gzip
import zipfile
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd
from connectome import Source, meta
from connectome.interface.nodes import Output, Silent

from .internals import checksum, register


@register(
    body_region='Head',
    license='CC BY-NC-SA 4.0',
    link='https://zenodo.org/record/6504722#.YsgwnNJByV4',
    modality=('MRI T1c', 'MRI T2hr'),
    prep_data_size=None,  # TODO: should be measured...
    raw_data_size='17G',
    task=('Segmentation', 'Classification', 'Domain Adaptation'),
)
@checksum('crossmoda2022')
class CrossMoDA(Source):
    """
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

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

    _root: str = None

    @meta
    def ids(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the locations of the zip archives')

        result = set()
        for archive in Path(_root).glob('*.zip'):
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

    @meta
    def train_source_df(_root):
        return pd.read_csv(Path(_root) / 'infos_source_training.csv', index_col='crossmoda_name')

    def _file(i, _root: Silent):
        for archive in Path(_root).glob('*.zip'):
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if i == '_'.join(Path(zipinfo.filename).stem.split('_')[:-1]) and 'Label' not in zipinfo.filename:
                        return zipfile.Path(archive, zipinfo.filename)

        raise ValueError(f'Id "{i}" not found')

    def image(_file) -> Union[np.ndarray, None]:
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                # most ct scans are integer-valued, this will help us improve compression rates
                #  (instead of using `image.get_fdata()`)
                return np.asarray(image.dataobj)

    def pixel_spacing(_file):
        """Returns pixel spacing along axes (x, y, z)"""
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return tuple(image.header['pixdim'][1:4])

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation"""
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    def split(_file) -> str:
        """The split in which this entry is contained: training_source, training_target, validation"""
        idx = int(_file.name.split('_')[2])
        dataset = _file.name.split('_')[1]

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

        raise ValueError(f'Cannot find split for the file: {_file}')

    def year(_file) -> int:
        """The year in which this entry was published: 2021 or 2022"""
        return int(_file.name[9:13])

    def masks(i, _file) -> Union[np.ndarray, None]:
        """Combined mask of schwannoma and cochlea (1 and 2 respectively)"""
        if 'T2' in _file.name:
            return None
        with (_file.parent / _file.name.replace('ceT1', 'Label')).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                mask = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return mask.get_fdata().astype(np.uint8)

    def koos_grade(i, train_source_df: Output, split: Output) -> int:
        """VS Tumour characteristic according to Koos grading scale: [1..4] or (-1 - post operative)"""
        if split != 'training_source':
            return None
        return (lambda x: -1 if x == 'post-operative-london' else int(x))(train_source_df.loc[i, 'koos'])
