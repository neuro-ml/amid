import gzip
import zipfile
from typing import Dict
from zipfile import ZipFile

import nibabel
import numpy as np

from .internals import Dataset, field, licenses, register


@register(
    body_region='Abdomen',
    license=licenses.CC_BY_40,
    link='https://zenodo.org/records/13767408',
    modality='CT',
    prep_data_size='30G',
    raw_data_size='30G',
    task='Abdominal organ pathologies segmentation',
)
class CURVAS(Dataset):
    """
    Pancreas, liver and kidney cysts segmentation from multi-rater annotated data.

    The dataset was used at the MICCAI 2024 CURVAS challenge.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download link: https://zenodo.org/records/13767408

    The `root` folder should contain the three downloaded .zip archives, namely:
    `training_set.zip`, `validation_set.zip` and `testing_set.zip`.

    Examples
    --------
    >>> # Place the downloaded folders in any folder and pass the path to the constructor:
    >>> ds = CURVAS(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 90
    >>> print(ds.image(ds.ids[5]).shape)
    # (512, 512, 1045)
    >>> print(ds.mask(ds.ids[35]).shape)
    # (512, 512, 992)

    """

    @property
    def ids(self):
        def _extract(split):
            archive = self.root / f'{split}_set.zip'
            with ZipFile(archive) as zf:
                namelist = [x for x in zf.namelist() if len(x.rstrip('/').split('/')) == 2]
                ids = [f'{x.split("/")[1]}-{split}' for x in namelist]
                return ids

        return sorted(
            [
                *_extract('training'),  # 20 Training   cases
                *_extract('validation'),  # 5  Validation cases
                *_extract('testing'),  # 65 Testing    cases
            ]
        )

    def _file(self, i, obj):
        uid, split = i.split('-')

        archive = self.root / f'{split}_set.zip'
        file = f'{split}_set/{uid}/{obj}.nii.gz'

        return zipfile.Path(archive, file)

    @field
    def image(self, i) -> np.ndarray:
        with self._file(i, 'image').open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.asarray(image.dataobj).astype(np.int16)

    @field
    def affine(self, i) -> np.ndarray:
        """The 4x4 matrix that gives the image's spatial orientation"""
        with self._file(i, 'image').open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    @field
    def masks(self, i) -> Dict[str, np.ndarray]:
        masks = {}
        for x in range(1, 4):
            with self._file(i, f'annotation_{x}').open('rb') as opened:
                with gzip.GzipFile(fileobj=opened) as nii:
                    nii = nibabel.FileHolder(fileobj=nii)
                    image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})

                    masks[f'annotation_{x}'] = np.asarray(image.dataobj).astype(np.uint8)

        return masks
