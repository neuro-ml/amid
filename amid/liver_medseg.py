import contextlib
import gzip
import re
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel as nb
import numpy as np

from .internals import Dataset, field, licenses, register


@register(
    body_region=('Chest', 'Abdomen'),
    license=licenses.CC_BYSA_40,
    link='https://www.medseg.ai/database/liver-segments-50-cases',
    modality='CT',
    prep_data_size='1,88G',
    raw_data_size='616M',
    task='Segmentation',
)
class LiverMedseg(Dataset):
    """
    LiverMedseg is a public CT segmentation dataset with 50 annotated images.
    Case collection of 50 livers with their segments.
    Images obtained from Decathlon Medical Segmentation competition

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download links:
    https://www.medseg.ai/database/liver-segments-50-cases

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = LiverMedseg(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 50
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 38)

    References
    ----------
    """

    @property
    def ids(self):
        result = set()
        with ZipFile(self.root / 'img.zip') as zf:
            for zipinfo in zf.infolist():
                if zipinfo.is_dir():
                    continue
                file_stem = Path(zipinfo.filename).stem
                result.add('liver_medseg_' + re.findall(r'\d+', file_stem)[0])

        return tuple(sorted(result))

    def _file(self, i):
        num_id = i.split('_')[-1]
        return zipfile.Path(self.root / 'img.zip', f'img{num_id}.nii.gz')

    @field
    def image(self, i) -> np.ndarray:
        with open_nii_gz_file(self._file(i)) as nii_file:
            return np.asarray(nii_file.dataobj)

    @field
    def affine(self, i) -> np.ndarray:
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz_file(self._file(i)) as nii_file:
            return nii_file.affine

    def spacing(self, i) -> tuple:
        with open_nii_gz_file(self._file(i)) as nii_file:
            return tuple(nii_file.header['pixdim'][1:4])

    @field
    def mask(self, i) -> np.ndarray:
        path = Path(str(self._file(i)).replace('img', 'mask'))
        folder, image = path.parent, path.name
        _file = zipfile.Path(folder, image)
        with open_nii_gz_file(_file) as nii_file:
            return np.asarray(nii_file.dataobj).astype(np.uint8)


# TODO: sync with amid.utils
@contextlib.contextmanager
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
