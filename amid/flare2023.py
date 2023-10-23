import gzip
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent
from py7zr import SevenZipFile

from .internals import checksum, register


@register(
    body_region='Abdomen',
    license=None,
    link='https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details-overview',
    modality='CT',
    prep_data_size='5G',
    raw_data_size='5G',
    task='Semi-supervised abdominal organ and tumors segmentation',
)
@checksum('flare2023')
class FLARE2023(Source):
    """
    An abdominal organ and tumor segmentation dataset for semi-supervised learning.

    The dataset was used at the MICCAI FLARE 2023 challenge.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    *NB!* Only validation cases are currently handeled for this particular dataset.

    Download link: https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details-dataset/

    The `root` folder should contain downloaded archives and files the same way they named and located in the cloud.

    Examples
    --------
    >>> # Place the downloaded folders in any folder and pass the path to the constructor:
    >>> ds = FLARE2023(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 100
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 55)
    >>> print(ds.mask(ds.ids[1]).shape)
    # (512, 512, 107)
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the locations of the downloaded data.')

        result = set()

        # 100 Validation cases (odd are labeled, even are not)
        archive = Path(_root) / 'validation100.7z'

        with SevenZipFile(archive) as szf:
            for file in szf.getnames():
                if not file.endswith('.nii.gz'):
                    continue

                i = file.split('_')[-2]
                prefix = 'VU' if int(i) % 2 == 0 else 'VL'

                result.add(prefix + i)

        return sorted(result)

    def _file(i, _root: Silent):
        # 100 Validation cases (odd are labeled, even are not)
        archive = Path(_root) / 'validation100.7z'

        if i.startswith('V'):
            with SevenZipFile(archive) as szf:
                file = f'validation/FLARE23Ts_{i[2:]}_0000.nii.gz'
                return szf.read(file)[file]

        raise ValueError(f'Id "{i}" not found')

    def image(_file):
        with gzip.GzipFile(fileobj=_file) as nii:
            nii = nibabel.FileHolder(fileobj=nii)
            image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
            return np.asarray(image.dataobj)

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation"""
        with gzip.GzipFile(fileobj=_file) as nii:
            nii = nibabel.FileHolder(fileobj=nii)
            image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
            return image.affine

    def mask(i, _root: Silent):
        if not i.startswith('V'):
            return None

        archive = list(Path(_root).glob('val-gt-50cases-for-sanity-check*'))
        assert len(archive) == 1, f'Multiple .zip archives detected {archive}. Expected only one'
        archive = archive[0]

        with ZipFile(archive) as zf:
            for file in zf.namelist():
                if i[2:] not in file:
                    continue

                with zipfile.Path(archive, file).open('rb') as opened:
                    with gzip.GzipFile(fileobj=opened) as nii:
                        nii = nibabel.FileHolder(fileobj=nii)
                        mask = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                        return np.asarray(mask.dataobj)
