import gzip
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel as nb
import numpy as np

from .internals import Dataset, field, licenses, register


@register(
    body_region='Head',
    license=licenses.CC_BYNCND_40,
    link='https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release',
    modality='MRI',
    prep_data_size='125G',
    raw_data_size='125G',
    task='Segmentation',
)
class HCP(Dataset):
    @property
    def ids(self):
        result = set()
        for archive in self.root.glob('*.zip'):
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue
                    result.add(zipinfo.filename.split('/')[0])

        return tuple(sorted(result))

    def _file(self, i):
        for archive in self.root.glob('*.zip'):
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue
                    file = Path(zipinfo.filename)
                    if (i in file.stem) and ('T1w_MPR1' in file.stem):
                        return zipfile.Path(str(archive), str(file))

    @field
    def image(self, i) -> np.ndarray:
        with self._file(i).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.int16(image.get_fdata())

    @field
    def affine(self, i) -> np.ndarray:
        with self._file(i).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    def spacing(self, i):
        with self._file(i).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return tuple(image.header['pixdim'][1:4])
