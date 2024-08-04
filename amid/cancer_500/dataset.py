import codecs
import json
import warnings
from functools import cached_property
from pathlib import Path

import numpy as np
import pydicom
from dicom_csv import (
    get_common_tag,
    get_orientation_matrix,
    get_pixel_spacing,
    get_slice_locations,
    get_tag,
    order_series,
    stack_images,
)
from dicom_csv.exceptions import TagMissingError
from tqdm.auto import tqdm

from ..internals import Dataset, field, register
from ..utils import get_series_date
from .nodules import get_nodules


@register(
    body_region='Thorax',
    modality='CT',
    task='Lung Cancer Detection',
    link='https://mosmed.ai/en/datasets/mosmeddata-kt-s-priznakami-raka-legkogo-tip-viii/',
    prep_data_size='103G',
    raw_data_size='187G',
)
class MoscowCancer500(Dataset):
    """
    The Moscow Radiology Cancer-500 dataset.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded files.
        If not provided, the cache is assumed to be already populated.


    Notes
    -----
    Download links:
    https://mosmed.ai/en/datasets/mosmeddata-kt-s-priznakami-raka-legkogo-tip-viii/
    After pressing the `download` button you will have to provide an email address to which further instructions
    will be sent.

    Examples
    --------
    >>> # Place the downloaded files in any folder and pass the path to the constructor:
    >>> ds = MoscowCancer500(root='/path/to/files/root')
    >>> print(len(ds.ids))
    # 979
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 67)
    """

    @cached_property
    def _mapping(self):
        path = self.root / 'series-to-files.json'
        if not path.exists():
            mapping = {}
            for file in tqdm(
                self.root.rglob('*'), total=sum(1 for _ in self.root.rglob('*')), desc='Analyzing folder structure'
            ):
                if file.is_dir():
                    continue

                series = pydicom.dcmread(file, specific_tags=[(0x0020, 0x000E)]).SeriesInstanceUID
                mapping[series].append(str(file.relative_to(self.root)))

            with open(path, 'w') as file:
                json.dump(mapping, file)
            return mapping

        with open(path) as file:
            return json.load(file)

    @property
    def ids(self):
        # this id has an undefined image orientation
        ignore = {'1.2.643.5.1.13.13.12.2.77.8252.604378326291403.583548115656123.'}
        return tuple(sorted(set(self._mapping) - ignore))

    def _series(self, i):
        series = [pydicom.dcmread(Path(self.root, 'dicom', f)) for f in self._mapping[i]]
        series = order_series(series, decreasing=False)
        return series

    @field
    def image(self, i):
        x = stack_images(self._series(i), -1).astype(np.int16)
        # DICOM specifies that the first 2 axes are (y, x). let's fix that
        return np.moveaxis(x, 0, 1)

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
        return get_pixel_spacing(self._series(i)).tolist()

    @field
    def slice_locations(self, i):
        return get_slice_locations(self._series(i))

    @field
    def orientation_matrix(self, i):
        return get_orientation_matrix(self._series(i))

    @field
    def instance_numbers(self, i):
        try:
            instance_numbers = [int(get_tag(i, 'InstanceNumber')) for i in self._series(i)]
            if not _is_monotonic(instance_numbers):
                warnings.warn('Ordered series has non-monotonic instance numbers.')

            return instance_numbers
        except TagMissingError:
            pass

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

    @field
    def nodules(self, i):
        folders = {Path(f).parent.name for f in self._mapping[i]}
        if len(folders) != 1:
            # can't determine protocol filename
            return

        (filename,) = folders
        protocol = json.load(codecs.open(str(self.root / 'protocols' / f'{filename}.json'), 'r', 'utf-8-sig'))

        series_number = get_common_tag(self._series(i), 'SeriesNumber')
        try:
            return get_nodules(protocol, series_number, self.slice_locations(i))
        except ValueError:
            pass


def _is_monotonic(sequence):
    sequence = list(sequence)
    return sequence == sorted(sequence) or sequence == sorted(sequence)[::-1]
