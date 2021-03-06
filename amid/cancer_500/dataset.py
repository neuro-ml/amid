import codecs
import datetime
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom
from connectome import Source, meta
from connectome.interface.nodes import Silent, Output
from dicom_csv import get_common_tag, order_series, get_tag, stack_images, get_pixel_spacing, get_slice_locations, \
    get_orientation_matrix
from dicom_csv.exceptions import TagMissingError, TagTypeError, ConsistencyError
from tqdm.auto import tqdm

from .nodules import get_nodules
from ..internals import checksum


@checksum('cancer_500')
class MoscowCancer500(Source):
    """
    The Moscow Radiology Cancer-500 dataset.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded files.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Download links:
    https://mosmed.ai/en/datasets/ct_lungcancer_500/

    Examples
    --------
    >>> # Place the downloaded files in any folder and pass the path to the constructor:
    >>> ds = MoscowCancer500(root='/path/to/files/root')
    >>> print(len(ds.ids))
    # 979
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 67)
    """

    _root: str = None

    @lru_cache(None)
    def _mapping(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the location of the downloaded files')

        _root = Path(_root)
        path = _root / 'series-to-files.json'
        if not path.exists():
            mapping = {}
            for file in tqdm(_root.rglob('*'), total=sum(1 for _ in _root.rglob('*')),
                             desc='Analyzing folder structure'):
                if file.is_dir():
                    continue

                series = pydicom.dcmread(file, specific_tags=[(0x0020, 0x000E)]).SeriesInstanceUID
                mapping[series].append(str(file.relative_to(_root)))

            with open(path, 'w') as file:
                json.dump(mapping, file)
            return mapping

        with open(path) as file:
            return json.load(file)

    @meta
    def ids(_mapping):
        # this id has an undefined image orientation
        ignore = {'1.2.643.5.1.13.13.12.2.77.8252.604378326291403.583548115656123.'}
        return tuple(sorted(set(_mapping) - ignore))

    def _series(i, _mapping, _root: Silent):
        series = [pydicom.dcmread(Path(_root, 'dicom', f)) for f in _mapping[i]]
        series = order_series(series)
        return series

    def image(_series):
        return stack_images(_series, -1).astype(np.int16)

    def study_uid(_series):
        return get_common_tag(_series, 'StudyInstanceUID')

    def series_uid(_series):
        return get_common_tag(_series, 'SeriesInstanceUID')

    def sop_uids(_series):
        return [str(get_tag(i, 'SOPInstanceUID')) for i in _series]

    def pixel_spacing(_series):
        return get_pixel_spacing(_series).tolist()

    def slice_locations(_series):
        return get_slice_locations(_series)

    def orientation_matrix(_series):
        return get_orientation_matrix(_series)

    def instance_numbers(_series):
        try:
            instance_numbers = [int(get_tag(i, 'InstanceNumber')) for i in _series]
            if not _is_monotonic(instance_numbers):
                warnings.warn(f'Ordered series has non-monotonic instance numbers.')

            return instance_numbers
        except TagMissingError:
            pass

    def conv_kernel(_series):
        return get_common_tag(_series, 'ConvolutionKernel', default=None)

    def kvp(_series):
        return get_common_tag(_series, 'KVP', default=None)

    def patient_id(_series):
        return get_common_tag(_series, 'PatientID', default=None)

    def study_date(_series):
        return _get_study_date(_series)

    def accession_number(_series):
        return get_common_tag(_series, 'AccessionNumber', default=None)

    def nodules(i, _mapping, _root: Silent, _series, slice_locations: Output):
        folders = {Path(f).parent.name for f in _mapping[i]}
        if len(folders) != 1:
            # can't determine protocol filename
            return

        filename, = folders
        protocol = json.load(codecs.open(str(Path(_root) / 'protocols' / f'{filename}.json'), 'r', 'utf-8-sig'))

        series_number = get_common_tag(_series, 'SeriesNumber')
        try:
            return get_nodules(protocol, series_number, slice_locations)
        except ValueError:
            pass


def _is_monotonic(sequence):
    sequence = list(sequence)
    return sequence == sorted(sequence) or sequence == sorted(sequence)[::-1]


def _get_study_date(series):
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

    if year < 1972:  # year of creation of first CT scanner
        return

    return datetime.date(year, month, day)
