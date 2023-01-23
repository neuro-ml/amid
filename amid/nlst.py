import json
from pathlib import Path

import numpy as np
import pydicom
from connectome import Source, meta
from connectome.interface.nodes import Silent
from dicom_csv import (
    Plane,
    drop_duplicated_slices,
    expand_volumetric,
    get_common_tag,
    get_orientation_matrix,
    get_pixel_spacing,
    get_slice_locations,
    get_slices_plane,
    get_tag,
    order_series,
    stack_images,
)

from .internals import checksum, licenses, register
from .utils import get_series_date


@register(
    body_region='Thorax',
    license=licenses.CC_BY_30,
    link='https://wiki.cancerimagingarchive.net/display/NLST/National+Lung+Screening+Trial',
    modality='CT',
    prep_data_size=None,  # TODO: should be measured...
    raw_data_size=None,  # TODO: should be measured...
    task=None,
)
@checksum('nlst')
class NLST(Source):
    """

        Dataset with low-dose CT scans of 26,254 patients acquired during National Lung Screening Trial.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder (usually called NLST) containing the patient subfolders (like 101426).
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Follow the download instructions at
    https://wiki.cancerimagingarchive.net/display/NLST/National+Lung+Screening+Trial.
    The dicoms should be placed under the following folders' structure:
        <...>/<NLST-root>/<patiend_id>/<study_uid>/<date>/<series_uid>/*.dcm

    Examples
    --------
    >>> ds = NLST(root='/path/to/NLST/')
    >>> print(len(ds.ids))
     ...
    >>> print(ds.image(ds.ids[0]).shape)
     ...
    >>> print(ds.mask(ds.ids[80]).shape)
     ...

    References
    ----------
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        return tuple(
            path.name
            for path in Path(_root).glob('*/*/*/*')
            if path.is_dir()
            if any(path.iterdir())
            if int(_load_json(path.parent / f'{path.name}.json')['Total'][5]) >= 8  # at least 8 slices
        )

    def _series(i, _root: Silent):
        (folder,) = Path(_root).glob(f'**/{i}')
        series = list(map(pydicom.dcmread, folder.iterdir()))
        series = expand_volumetric(series)
        assert get_common_tag(series, 'Modality') == 'CT'
        assert get_slices_plane(series) == Plane.Axial
        series = drop_duplicated_slices(series)
        series = order_series(series, decreasing=False)
        return series

    def image(_series):
        return np.moveaxis(stack_images(_series, -1).astype(np.int16), 0, 1)

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

    def conv_kernel(_series):
        return get_common_tag(_series, 'ConvolutionKernel', default=None)

    def kvp(_series):
        return get_common_tag(_series, 'KVP', default=None)

    def patient_id(_series):
        return get_common_tag(_series, 'PatientID', default=None)

    def study_date(_series):
        return get_series_date(_series)

    def accession_number(_series):
        return get_common_tag(_series, 'AccessionNumber', default=None)


def _load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
