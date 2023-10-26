import gzip
import json
import zipfile
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import nibabel
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from .internals import licenses, normalize


class VerSeBase(Source):
    """
    A Vertebral Segmentation Dataset with Fracture Grading [1]_

    The dataset was used in the MICCAI-2019 and MICCAI-2020 Vertebrae Segmentation Challenges.

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
        2019: https://osf.io/jtfa5/
        2020: https://osf.io/4skx2/

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = VerSe(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 374
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 214)

    References
    ----------
    .. [1] Löffler MT, Sekuboyina A, Jacob A, et al. A Vertebral Segmentation Dataset with Fracture Grading.
       Radiol Artif Intell. 2020;2(4):e190138. Published 2020 Jul 29. doi:10.1148/ryai.2020190138
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the locations of the zip archives')

        result = set()
        for archive in Path(_root).glob('*.zip'):
            with ZipFile(archive) as zf:
                for file in zf.namelist():
                    if '/rawdata/' not in file:
                        continue

                    file = Path(file)
                    patient = file.parent.name[4:]
                    name = file.name
                    if 'split' in name:
                        i = name.split('split')[1][1:]
                        i = i.split('_')[0]
                    else:
                        i = patient

                    assert i not in result, i
                    result.add(i)

        return sorted(result)

    def _file(i, _root: Silent):
        for archive in Path(_root).glob('*.zip'):
            with ZipFile(archive) as zf:
                for file in zf.namelist():
                    if '/rawdata/' in file and i in file:
                        return zipfile.Path(archive, file)

        raise ValueError(f'Id "{i}" not found')

    def image(_file):
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                # most ct scans are integer-valued, this will help us improve compression rates
                #  (instead of using `image.get_fdata()`)
                return np.asarray(image.dataobj)

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation"""
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    def split(_file):
        """The split in which this entry is contained: training, validate, test"""
        # it's ugly, but it gets the job done (;
        return _file.parent.parent.parent.name.split('_')[-1].split('9')[-1]

    def patient(_file):
        """The unique patient id"""
        return _file.parent.name[4:]

    def year(_file):
        """The year in which this entry was published: 2019, 2020"""
        year = _file.parent.parent.parent.name
        if year.startswith('dataset-verse'):
            assert '19' in year
            return 2019
        return 2020

    def _derivatives(_file):
        return _file.parent.parent.parent / 'derivatives' / _file.parent.name

    def centers(i, _derivatives):
        """Vertebrae centers in format {label: [x, y, z]}"""
        ann = [f for f in _derivatives.iterdir() if f.name.endswith('.json') and i in f.name]
        if not ann:
            return {}
        assert len(ann) == 1
        (ann,) = ann

        with ann.open() as file:
            ann = json.load(file)

        return {k['label']: [k['X'], k['Y'], k['Z']] for k in ann[1:]}

    def masks(i, _derivatives) -> Union[np.ndarray, None]:
        """Vertebrae masks"""
        ann = [f for f in _derivatives.iterdir() if f.name.endswith('.nii.gz') and i in f.name]
        if not ann:
            return
        assert len(ann) == 1
        (ann,) = ann

        with ann.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                mask = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return mask.get_fdata().astype(np.uint8)


VerSe = normalize(
    VerSeBase,
    'VerSe',
    'verse',
    body_region=('Thorax', 'Abdomen'),
    modality='CT',
    task='Vertebrae Segmentation',
    link='https://osf.io/4skx2/',
    raw_data_size='97G',
    license=licenses.CC_BYSA_40,
)
