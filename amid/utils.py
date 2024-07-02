import contextlib
import datetime
import functools
import itertools
import zipfile
from gzip import GzipFile
from os import PathLike
from pathlib import Path
from typing import List, Union

import nibabel
import numpy as np
from dicom_csv import get_common_tag, order_series, stack_images
from dicom_csv.exceptions import ConsistencyError, TagTypeError
from pydicom import Dataset, dcmread


Numeric = Union[float, int]
PathOrStr = Union[str, PathLike]


@contextlib.contextmanager
def unpack(root: PathOrStr, relative: str, archive_root_name: str = None, archive_ext: str = None):
    """Provides the absolute path to the file in both scenarios: inside archive or inside folder.

    Parameters
    ----------
    root : str, Path
        Absolute path to the downloaded archive or the unpacked archive root.
    relative : str, Path
        Relative file path inside the archive. Archive's root folder sholud be ommited.
    archive_root_name : str, Path, optional
        If `root` is a archive, it's root folder name shold be given.
    archive_ext: {'.zip'}, optional
        Compression algorithm used to create the archive

    Returns
    -------
    unpacked : Path
        Absolute file path to be opened.
    is_unpacked : {True, False}
        Reached file state. `True` if the file is located inside archive, `False` otherwise.
    """
    unpacked = Path(root) / relative

    if unpacked.exists():
        yield unpacked, True
    elif archive_ext == '.zip':
        with zipfile.Path(root, str(Path(archive_root_name, relative))).open('rb') as unpacked:
            yield unpacked, False
    else:
        raise ValueError('Unexpected file path or unsupported compression algorithm.')


@contextlib.contextmanager
def open_nii_gz_file(unpacked):
    """Opens ``.nii.gz`` file if it is packed in archive

    Examples
    --------
    >>> with unpack('/path/to/archive.zip', 'relative/file/path', 'root', '.zip') as (unpacked, is_unpacked):
    >>>     with open_nii_gz_file(unpacked) as image:
    >>>         print(np.asarray(image.dataobj).shape)
    # (512, 512, 256)
    """
    with GzipFile(fileobj=unpacked) as nii:
        nii = nibabel.FileHolder(fileobj=nii)
        yield nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})


def get_series_date(series):
    try:
        study_date = get_common_tag(series, 'StudyDate')
    except (TagTypeError, ConsistencyError):
        return

    if not isinstance(study_date, str) or not study_date.isnumeric() or len(study_date) != 8:
        return

    try:
        year = int(study_date[:4])
        month = int(study_date[4:6])
        day = int(study_date[6:])
    except TypeError:
        return

    if year < 1972:  # the year of creation of the first CT scanner
        return

    return datetime.date(year, month, day)


def propagate_none(func):
    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        return None if (x is None) else func(x, *args, **kwargs)

    return wrapper


def deprecate(message=None):
    def decorator(func):
        return functools.wraps(func)(np.deprecate(message=message)(func))

    return decorator


def image_from_dicom_folder(folder: Union[str, Path]) -> np.ndarray:
    return stack_images(series_from_dicom_folder(folder))


def series_from_dicom_folder(folder: Union[str, Path]) -> List[Dataset]:
    return order_series([dcmread(p) for p in Path(folder).glob('*.dcm')])


# TODO: stolen from dpipe for now
def mask_to_box(mask: np.ndarray):
    """
    Find the smallest box that contains all true values of the ``mask``.
    """
    if not mask.any():
        raise ValueError('The mask is empty.')

    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0
        start.insert(0, left)
        stop.insert(0, right + 1)
    return start, stop
