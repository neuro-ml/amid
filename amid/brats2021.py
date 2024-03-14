import contextlib
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent

from .internals import licenses, normalize
from .utils import open_nii_gz_file, unpack


class BraTS2021Base(Source):
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

    _root: str = None

    def _base(_root: Silent) -> Path:
        if _root is None:
            raise ValueError('Please pass the locations of the zip archives')
        return Path(_root)

    @meta
    def ids(_base):
        return sorted(_get_ids_or_file(_base, 'TrainingData') + _get_ids_or_file(_base, 'ValidationData'))

    def fold(i, _base):
        return 'ValidationData' if _get_ids_or_file(_base, 'ValidationData', check_id=i) else 'TrainingData'

    @meta
    def mapping21_17(_base) -> pd.DataFrame:
        return pd.read_csv(_base / 'BraTS21-17_Mapping.csv')

    def subject_id(i) -> str:
        return i.rsplit('_', 1)[0]

    def modality(i) -> str:
        return i.rsplit('_', 1)[1]

    def image(i, _base, fold: Output):
        root, relative = _get_ids_or_file(_base, fold, check_id=i, return_image=True)
        with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
            return np.asarray(nii_image.dataobj)

    def mask(i, _base, fold: Output):
        if fold == 'ValidationData':
            return None
        else:
            root, relative = _get_ids_or_file(_base, fold, check_id=i, return_segm=True)
            with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
                return np.asarray(nii_image.dataobj)

    def spacing(i, _base, fold: Output):
        """Returns the voxel spacing along axes (x, y, z)."""
        root, relative = _get_ids_or_file(_base, fold, check_id=i, return_image=True)
        with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
            return tuple(nii_image.header['pixdim'][1:4])

    def affine(i, _base, fold: Output):
        """Returns 4x4 matrix that gives the image's spatial orientation."""
        root, relative = _get_ids_or_file(_base, fold, check_id=i, return_image=True)
        with _load_nibabel_probably_from_zip(root, relative, '.', '.zip') as nii_image:
            return nii_image.affine


BraTS2021 = normalize(
    BraTS2021Base,
    'BraTS2021',
    'brats2021',
    body_region='Head',
    license=licenses.CC_BYNCSA_40,
    link='http://www.braintumorsegmentation.org/',
    modality=('MRI T1', 'MRI T1Gd', 'MRI T2', 'MRI T2-FLAIR'),
    prep_data_size='8,96G',
    raw_data_size='15G',
    task=('Segmentation', 'Classification', 'Domain Adaptation'),
    ignore=['mapping21_17'],
)


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
