import gzip
from typing import Union

import nibabel
import numpy as np

from .internals import Dataset, field, register


@register(
    body_region='Thorax',
    modality='CT',
    task='COVID-19 Segmentation',
    link='https://mosmed.ai/en/datasets/covid191110/',
    raw_data_size='21G',
)
class MoscowCovid1110(Dataset):
    """
    The Moscow Radiology COVID-19 dataset.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded files.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download links:
    https://mosmed.ai/en/datasets/covid191110/

    Examples
    --------
    >>> # Place the downloaded files in any folder and pass the path to the constructor:
    >>> ds = MoscowCovid1110(root='/path/to/files/root')
    >>> print(len(ds.ids))
    # 1110
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 43)
    """

    @property
    def ids(self):
        return sorted({f.name[:-7] for f in self.root.glob('CT-*/*')})

    def _file(self, i):
        return next(self.root.glob(f'CT-*/{i}.nii.gz'))

    @field
    def image(self, i) -> np.ndarray:
        with self._file(i).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                # most ct scans are integer-valued, this will help us improve compression rates
                #  (instead of using `image.get_fdata()`)
                return np.asarray(image.dataobj)

    @field
    def affine(self, i) -> np.ndarray:
        with self._file(i).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    @field
    def label(self, i) -> str:
        return self._file(i).parent.name[3:]

    @field
    def mask(self, i) -> Union[np.ndarray, None]:
        path = self.root / 'masks' / f'{i}_mask.nii.gz'
        if not path.exists():
            return

        with path.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.asarray(image.dataobj) > 0.5
