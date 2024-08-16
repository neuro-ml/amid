import deli
import numpy as np
import pydicom
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
from tqdm.auto import tqdm

from .internals import Dataset, field, licenses, register
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
class NLST(Dataset):
    """

        Dataset with low-dose CT scans of 26,254 patients acquired during National Lung Screening Trial.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder (usually called NLST) containing the patient subfolders (like 101426).
        If not provided, the cache is assumed to be already populated.

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

    @property
    def ids(self):
        ids = []
        for path in tqdm(list(self.root.iterdir())):
            series_uid2num_slices = {p.stem: int(deli.load(p)['Total'][5]) for p in path.glob('*/*/*.json')}
            ids.append(max(series_uid2num_slices, key=series_uid2num_slices.get))

        return ids

    def _series(self, i):
        (folder,) = self.root.glob(f'**/{i}')
        series = list(map(pydicom.dcmread, folder.iterdir()))
        series = expand_volumetric(series)
        assert get_common_tag(series, 'Modality') == 'CT'
        assert get_slices_plane(series) == Plane.Axial
        series = drop_duplicated_slices(series)
        series = order_series(series, decreasing=False)
        return series

    @field
    def image(self, i):
        return np.moveaxis(stack_images(self._series(i), -1).astype(np.int16), 0, 1)

    @field
    def study_uid(self, i):
        return get_common_tag(self._series(i), 'StudyInstanceUID')

    @field
    def series_uid(self, i):
        return get_common_tag(self._series(i), 'SeriesInstanceUID')

    @field
    def sop_uids(self, i):
        return [str(get_tag(i, 'SOPInstanceUID')) for i in self._series(i)]

    @field
    def pixel_spacing(self, i):
        return get_pixel_spacing(self, i).tolist()

    @field
    def slice_locations(self, i):
        return get_slice_locations(self, i)

    @field
    def orientation_matrix(self, i):
        return get_orientation_matrix(self, i)

    @field
    def conv_kernel(self, i):
        return get_common_tag(self._series(i), 'ConvolutionKernel', default=None)

    @field
    def kvp(self, i):
        return get_common_tag(self._series(i), 'KVP', default=None)

    @field
    def patient_id(self, i):
        return get_common_tag(self._series(i), 'PatientID', default=None)

    @field
    def study_date(self, i):
        return get_series_date(self._series(i))

    @field
    def accession_number(self, i):
        return get_common_tag(self._series(i), 'AccessionNumber', default=None)
