import contextlib
import gzip
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel as nb
import numpy as np

from .internals import Dataset, field, register


@register(
    body_region=('Head', 'Abdominal'),
    license=None,  # FIXME: inherit licenses from the original datasets...
    link='http://medicalood.dkfz.de/web/',
    modality=('MRI', 'CT'),
    prep_data_size='405G',
    raw_data_size='120G',
    task='Out-of-distribution detection',
)
class MOOD(Dataset):
    """
    A (M)edival (O)ut-(O)f-(D)istribution analysis challenge [1]_

    This dataset contains raw brain MRI and abdominal CT images.

    Number of training samples:
    - Brain: 800 scans ( 256 x 256 x 256 )
    - Abdominal: 550 scans ( 512 x 512 x 512 )

    For each setup there are 4 toy test samples with OOD cases.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Follow the download instructions at https://www.synapse.org/#!Synapse:syn21343101/wiki/599515.

    Then, the folder with raw downloaded data should contain four zip archives with data
    (`abdom_toy.zip`, `abdom_train.zip`, `brain_toy.zip` and `brain_train.zip`).

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = MOOD(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 1358
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 512)
    >>> print(ds.pixel_label(ds.ids[0]).shape)
    # (512, 512, 512)

    References
    ----------
    .. [1] Zimmerer, Petersen, et al. "Medical Out-of-Distribution Analysis Challenge 2022."
           doi: 10.5281/zenodo.6362313 (2022).
    """

    @property
    def ids(self):
        result = set()
        # zip archives for train images:
        for archive in self.root.glob('*.zip'):
            if 'brain' in str(archive):  # define whether it is brain (MRI) or abdominal (CT)
                task = 'brain'
            else:
                task = 'abdom'

            if 'toy' in str(archive):  # fold - train or toy test
                fold = 'toy'
            else:
                fold = 'train'

            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue

                    file_stem = Path(zipinfo.filename).stem
                    if '.nii' in file_stem:
                        if fold == 'train':
                            result.add(f'mood_{task}_{fold}_{file_stem.split(".nii")[0]}')
                        # fold == 'toy'
                        else:
                            result.add(f'mood_{task}_{file_stem.split(".nii")[0]}')

        return tuple(sorted(result))

    @field
    def fold(self, i):
        """Returns fold: train or toy (test)."""
        if 'train' in i:
            return 'train'
        # if 'toy' in i:
        return 'toy'

    @field
    def task(self, i):
        """Returns task: brain (MRI) or abdominal (CT)."""
        if 'brain' in i:
            return 'brain'
        # if 'abdom' in i:
        return 'abdom'

    def _file(self, i):
        task, fold, num_id = i.split('_')[-3:]
        if fold == 'train':
            return zipfile.Path(self.root / f'{task}_{fold}.zip', f'{task}_{fold}/{num_id}.nii.gz')
        return zipfile.Path(self.root / f'{task}_{fold}.zip', f'toy/toy_{num_id}.nii.gz')

    @field
    def image(self, i):
        with open_nii_gz_file(self._file(i)) as nii_image:
            return np.asarray(nii_image.dataobj)

    @field
    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz_file(self._file(i)) as nii_image:
            return nii_image.affine

    def spacing(self, i):
        """Returns voxel spacing along axes (x, y, z)."""
        with open_nii_gz_file(self._file(i)) as nii_image:
            return tuple(nii_image.header['pixdim'][1:4])

    @field
    def sample_label(self, i):
        """
        Returns sample-level OOD score for toy examples and None otherwise.
        0 indicates no abnormality and 1 indicates abnormal input.
        """
        file = self._file(i)
        if 'toy' in file.name:
            with (file.parent.parent / 'toy_label/sample' / f'{file.name}.txt').open('r') as nii:
                return int(nii.read())

    @field
    def pixel_label(self, i):
        """
        Returns voxel-level OOD scores for toy examples and None otherwise.
        0 indicates no abnormality and 1 indicates abnormal input.
        """
        file = self._file(i)
        if 'toy' in file.name:
            with open_nii_gz_file(file.parent.parent / 'toy_label/pixel' / file.name) as nii_image:
                return np.bool_(nii_image.get_fdata())


# TODO: sync with amid.utils
@contextlib.contextmanager
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
