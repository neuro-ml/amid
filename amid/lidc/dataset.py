import datetime
import os
from typing import List, Tuple, Union

import numpy as np
import pylidc as pl
from dicom_csv import (
    Series,
    expand_volumetric,
    get_common_tag,
    get_orientation_matrix,
    get_tag,
    order_series,
    stack_images,
)
from pylidc.utils import consensus
from scipy import stats

from ..internals import Dataset, field, licenses, register
from ..utils import PathOrStr, get_series_date
from .nodules import get_nodule
from .typing import LIDCNodule


@register(
    body_region='Chest',
    license=licenses.CC_BY_30,
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254',
    modality='CT',
    prep_data_size='71,2G',
    raw_data_size='126G',
    task='Lung nodules segmentation',
)
class LIDC(Dataset):
    """
    The (L)ung (I)mage (D)atabase (C)onsortium image collection (LIDC-IDRI) [1]_
    consists of diagnostic and lung cancer screening thoracic computed tomography (CT) scans
    with marked-up annotated lesions and lung nodules segmentation task.
    Scans contains multiple expert annotations.

    Number of CT scans: 1018.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Follow the download instructions at https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254.

    Then, the folder with raw downloaded data should contain folder `LIDC-IDRI`,
    which contains folders `LIDC-IDRI-*`.

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = LIDC(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 1018
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 194)
    >>> print(ds.cancer(ds.ids[0]).shape)
    # (512, 512, 194)

    References
    ----------
    .. [1] Armato III, McLennan, et al. "The lung image database consortium (lidc) and image database
    resource initiative (idri): a completed reference database of lung nodules on ct scans."
    Medical physics 38(2) (2011): 915–931.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041807/
    """

    def __init__(self, root: PathOrStr):
        super().__init__(root)
        self._check_config()

    def _check_config(self):
        pylidc_config_start = '[dicom]\npath = '
        if os.path.exists(os.path.expanduser('~/.pylidcrc')):
            with open(os.path.expanduser('~/.pylidcrc'), 'r') as config_file:
                content = config_file.read()
            if content == f'{pylidc_config_start}{self.root}':
                return

        # save _root path to ~/.pylidcrc file for pylidc
        with open(os.path.expanduser('~/.pylidcrc'), 'w') as config_file:
            config_file.write(f'{pylidc_config_start}{self.root}')

    @property
    def ids(self):
        result = [scan.series_instance_uid for scan in pl.query(pl.Scan).all()]
        return tuple(sorted(result))

    def _scan(self, i) -> pl.Scan:
        _id = i.split('_')[-1]
        return pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == _id).first()

    def _series(self, i) -> Series:
        series = expand_volumetric(self._scan(i).load_all_dicom_images(verbose=False))
        series = order_series(series)
        return series

    def _shape(self, i) -> Tuple[int, int, int]:
        return stack_images(self._series(i), -1).shape

    @field
    def image(self, i) -> np.ndarray:
        return self._scan(i).to_volume(verbose=False)

    @field
    def study_uid(self, i) -> str:
        return self._scan(i).study_instance_uid

    @field
    def series_uid(self, i) -> str:
        return self._scan(i).series_instance_uid

    @field
    def patient_id(self, i) -> str:
        return self._scan(i).patient_id

    @field
    def sop_uids(self, i) -> List[str]:
        return [str(get_tag(i, 'SOPInstanceUID')) for i in self._series(i)]

    @field
    def pixel_spacing(self, i) -> List[float]:
        spacing = self._scan(i).pixel_spacing
        return [spacing, spacing]

    @field
    def slice_locations(self, i) -> np.ndarray:
        return self._scan(i).slice_zvals

    # @field
    def spacing(self, i) -> Tuple[float, float, float]:
        """
        Volumetric spacing of the image.
        The maximum relative difference in `slice_locations` < 1e-3
        (except 4 images listed below),
        so we allow ourselves to use the common spacing for the whole 3D image.

        Note
        ----
        The `slice_locations` attribute typically (but not always!) has the constant step.
        In LIDC dataset, only 4 images have difference in `slice_locations` > 1e-3:
            1.3.6.1.4.1.14519.5.2.1.6279.6001.526570782606728516388531252230
            1.3.6.1.4.1.14519.5.2.1.6279.6001.329334252028672866365623335798
            1.3.6.1.4.1.14519.5.2.1.6279.6001.245181799370098278918756923992
            1.3.6.1.4.1.14519.5.2.1.6279.6001.103115201714075993579787468219
        And these differences appear in the maximum of 3 slices.
        Therefore, we consider their impact negligible.
        """
        return (*self.pixel_spacing(i), stats.mode(np.diff(self.slice_locations(i)))[0].item())

    @field
    def contrast_used(self, i) -> bool:
        """If the DICOM file for the scan had any Contrast tag, this is marked as `True`."""
        return self._scan(i).contrast_used

    @field
    def is_from_initial(self, i) -> bool:
        """
        Indicates whether or not this PatientID was tagged as
        part of the initial 399 release.
        """
        return self._scan(i).is_from_initial

    @field
    def orientation_matrix(self, i) -> np.ndarray:
        return get_orientation_matrix(self._series(i))

    @field
    def sex(self, i) -> Union[str, None]:
        return get_common_tag(self._series(i), 'PatientSex', default=None)

    @field
    def age(self, i) -> Union[str, None]:
        return get_common_tag(self._series(i), 'PatientAge', default=None)

    @field
    def conv_kernel(self, i) -> Union[str, None]:
        return get_common_tag(self._series(i), 'ConvolutionKernel', default=None)

    @field
    def kvp(self, i) -> Union[str, None]:
        return get_common_tag(self._series(i), 'KVP', default=None)

    @field
    def tube_current(self, i) -> Union[str, None]:
        return get_common_tag(self._series(i), 'XRayTubeCurrent', default=None)

    @field
    def study_date(self, i) -> Union[datetime.date, None]:
        return get_series_date(self._series(i))

    @field
    def accession_number(self, i) -> Union[str, None]:
        return get_common_tag(self._series(i), 'AccessionNumber', default=None)

    @field
    def nodules(self, i) -> List[List[LIDCNodule]]:
        nodules = []
        for anns in self._scan(i).cluster_annotations():
            nodule_annotations = []
            for ann in anns:
                nodule_annotations.append(get_nodule(ann))
            nodules.append(nodule_annotations)
        return nodules

    @field
    def nodules_masks(self, i) -> List[List[np.ndarray]]:
        nodules = []
        for anns in self._scan(i).cluster_annotations():
            nodule_annotations = []
            for ann in anns:
                nodule_annotations.append(ann.boolean_mask())
            nodules.append(nodule_annotations)
        return nodules

    @field
    def cancer(self, i) -> np.ndarray:
        cancer = np.zeros(self._shape(i), dtype=bool)
        for anns in self._scan(i).cluster_annotations():
            cancer |= consensus(anns, pad=np.inf)[0]

        return cancer
