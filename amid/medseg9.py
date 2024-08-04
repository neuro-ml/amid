import contextlib
import gzip
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel as nb
import numpy as np

from .internals import Dataset, field, licenses, register


@register(
    body_region='Chest',
    license=licenses.CC0_10,
    link='http://medicalsegmentation.com/covid19/',
    modality='CT',
    prep_data_size='300M',
    raw_data_size='310M',
    task='COVID-19 segmentation',
)
class Medseg9(Dataset):
    """

    Medseg9 is a public COVID-19 CT segmentation dataset with 9 annotated images.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Data can be downloaded here: http://medicalsegmentation.com/covid19/.

    Then, the folder with raw downloaded data should contain three zip archives with data and masks
    (`rp_im.zip`, `rp_lung_msk.zip`, `rp_msk.zip`).

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = Medseg9(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 9
    >>> print(ds.image(ds.ids[0]).shape)
    # (630, 630, 45)
    >>> print(ds.covid(ds.ids[0]).shape)
    # (630, 630, 45)

    """

    @property
    def ids(self):
        result = set()

        with ZipFile(self.root / 'rp_msk.zip') as zf:
            for zipinfo in zf.infolist():
                if zipinfo.is_dir():
                    continue
                file_stem = Path(zipinfo.filename).stem
                result.add('medseg9_' + file_stem.split('.nii')[0])

        return tuple(sorted(result))

    @staticmethod
    def _filename(i):
        num_id = i.split('_')[-1]
        return f'{num_id}.nii.gz'

    def _file(self, i):
        return zipfile.Path(self.root / 'rp_im.zip', f'rp_im/{self._filename(i)}')

    @field
    def image(self, i):
        with open_nii_gz_file(self._file(i)) as nii_image:
            # most CT/MRI scans are integer-valued, this will help us improve compression rates
            return np.int16(nii_image.get_fdata())

    @field
    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz_file(self._file(i)) as nii_image:
            return nii_image.affine

    @field
    def lungs(self, i):
        mask_file = zipfile.Path(self.root / 'rp_lung_msk.zip', f'rp_lung_msk/{self._filename(i)}')
        with open_nii_gz_file(mask_file) as nii_image:
            return np.bool_(nii_image.get_fdata())

    @field
    def covid(self, i):
        """
        int16 mask.
        0 - normal, 1 - ground-glass opacities (матовое стекло), 2 - consolidation (консолидация).
        """
        mask_file = zipfile.Path(self.root / 'rp_msk.zip', f'rp_msk/{self._filename(i)}')
        with open_nii_gz_file(mask_file) as nii_image:
            # most CT/MRI scans are integer-valued, this will help us improve compression rates
            return np.uint8(nii_image.get_fdata())


# TODO: sync with amid.utils
@contextlib.contextmanager
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
