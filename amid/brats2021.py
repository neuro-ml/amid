import contextlib
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd

from .internals import Dataset, field, licenses, register
from .utils import open_nii_gz_file, unpack


@register(
    body_region='Head',
    license=licenses.CC_BYNCSA_40,
    link='http://www.braintumorsegmentation.org/',
    modality=('MRI T1', 'MRI T1Gd', 'MRI T2', 'MRI T2-FLAIR'),
    prep_data_size='8,96G',
    raw_data_size='15G',
    task=('Segmentation', 'Classification', 'Domain Adaptation'),
)
class BraTS2021(Dataset):
    """
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download links:
    2021: http://www.braintumorsegmentation.org/

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = BraTS2021(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 5880
    >>> print(ds.image(ds.ids[0]).shape)
    # (240, 240, 155)

    References
    ----------
    """

    @property
    def ids(self):
        return sorted(_get_ids_or_file(self.root, 'TrainingData') + _get_ids_or_file(self.root, 'ValidationData'))

    @field
    def fold(self, i) -> str:
        return 'ValidationData' if _get_ids_or_file(self.root, 'ValidationData', check_id=i) else 'TrainingData'

    @property
    def mapping21_17(self) -> pd.DataFrame:
        return pd.read_csv(self.root / 'BraTS21-17_Mapping.csv')

    @field
    def subject_id(self, i) -> str:
        return i.rsplit('_', 1)[0]

    @field
    def modality(self, i) -> str:
        return i.rsplit('_', 1)[1]

    @field
    def image(self, i) -> np.ndarray:
        root, relative = _get_ids_or_file(self.root, self.fold(i), check_id=i, return_image=True)
        with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
            return np.asarray(nii_image.dataobj)

    def mask(self, i) -> Union[np.ndarray, None]:
        if self.fold(i) == 'ValidationData':
            return None
        else:
            root, relative = _get_ids_or_file(self.root, self.fold(i), check_id=i, return_segm=True)
            with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
                return np.asarray(nii_image.dataobj)

    def spacing(self, i):
        """Returns the voxel spacing along axes (x, y, z)."""
        root, relative = _get_ids_or_file(self.root, self.fold(i), check_id=i, return_image=True)
        with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
            return tuple(nii_image.header['pixdim'][1:4])

    @field
    def affine(self, i) -> np.ndarray:
        """Returns 4x4 matrix that gives the image's spatial orientation."""
        root, relative = _get_ids_or_file(self.root, self.fold(i), check_id=i, return_image=True)
        with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
            return nii_image.affine


def _get_ids_or_file(
    base_path,
    archive_name_part: str = 'TrainingData',
    check_id: str = None,
    return_image: bool = False,
    return_segm: bool = False,
):
    # TODO: implement the same functionality for folder extraction.
    ids = []
    for archive in base_path.glob('*.zip'):
        if archive_name_part in archive.name:
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if not zipinfo.is_dir():
                        file = Path(zipinfo.filename)
                        _id = file.stem.replace('.nii', '')

                        if 'seg' not in _id:
                            ids.append(_id)

                        if (check_id is not None) and (check_id == _id):
                            if return_segm:
                                return str(archive), str(file)[: -len('.nii.gz')].rsplit('_', 1)[0] + '_seg.nii.gz'

                            if return_image:
                                return str(archive), str(file)

                            return True  # if check_id in archive

    return ids if (check_id is None) else False  # if check_id not in archive


@contextlib.contextmanager
def _load_nibabel_probably_from_zip(root: str, relative: str, archive_root_name: str = None, archive_ext: str = None):
    with unpack(root, relative, archive_root_name, archive_ext) as (unpacked, is_unpacked):
        if is_unpacked:
            yield nibabel.load(unpacked)
        else:
            with open_nii_gz_file(unpacked) as nii_image:
                yield nii_image
