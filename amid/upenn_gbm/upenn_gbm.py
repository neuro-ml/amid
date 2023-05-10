from functools import lru_cache
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent

from ..internals import checksum, licenses, register
from .data_classes import AcquisitionInfo, ClinicalInfo


@register(
    body_region='Head',
    license=licenses.CC_BY_40,
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642',
    modality=('FLAIR', 'MRI T1', 'MRI T1GD', 'MRI T2', 'DSC MRI', 'DTI MRI'),
    prep_data_size='70G',
    raw_data_size='69G',
    task='Segmentation',
)
@checksum('upenn_gbm')
class UPENN_GBM(Source):
    """
    Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma
      (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM).
    Dataset contains 630 patients.

    All samples are registered to a common atlas (SRI)
        using a uniform preprocessing and the segmentation are aligned with them.


    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Follow the download instructions at https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642
    Download to the root folder nifti images and metadata. Organise folder as folows:


    <...>/<UPENN-root>/NIfTI-files/images_segm/UPENN-GBM-00054_11_segm.nii.gz
    <...>/<UPENN-root>/NIfTI-files/...

    <...>/<UPENN-root>/UPENN-GBM_clinical_info_v1.0.csv
    <...>/<UPENN-root>/UPENN-GBM_acquisition.csv


    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = UPENN_GBM(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 671
    >>> print(ds.image(ds.ids[215]).shape)
    # (4, 240, 240, 155)
    >>> print(d.acqusition_info(d.ids[215]).manufacturer)
    # SIEMENS

    References
    ----------
    .. [1] Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie,
      J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S.,
    Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., … Davatzikos,
    C. (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo
    Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM)
    (Version 2) [Data set]. The Cancer Imaging Archive.
    https://doi.org/10.7937/TCIA.709X-DN49

    """

    _root: str = None

    def _base(_root: Silent):
        if _root is None:
            raise ValueError('Please provide the `root` argument')
        return Path(_root)

    @meta
    def ids(_base):
        ids = [x.name for x in (_base / 'NIfTI-files/images_structural').iterdir()]
        return tuple(sorted(ids))

    def modalities():
        return ['T1', 'T1GD', 'T2', 'FLAIR']

    def dsc_modalities():
        return ['', 'ap-rCBV', 'PH', 'PSR']

    def dti_modalities():
        return ['AD', 'FA', 'RD', 'TR']

    def _mask_path(i, _base):
        p1 = _base / 'NIfTI-files/images_segm'
        p2 = _base / 'NIfTI-files/automated_segm'
        p1 = list(p1.glob(i + '*'))
        p2 = list(p2.glob(i + '*'))
        return p1[0] if p1 else p2[0] if p2 else None

    def mask(_mask_path: Path):
        if not _mask_path:
            return None
        return np.asarray(nb.load(_mask_path).get_fdata())

    def is_mask_automated(_mask_path: Path):
        if _mask_path is None:
            return None
        return _mask_path.parent.name == 'automated_segm'

    def image(i, modalities: Output, _base):
        path = _base / f'NIfTI-files/images_structural/{i}'
        image_pathes = [path / f'{i}_{mod}.nii.gz' for mod in modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return np.stack(images)

    def image_unstripped(i, modalities: Output, _base):
        path = _base / f'NIfTI-files/images_structural_unstripped/{i}'
        image_pathes = [path / f'{i}_{mod}_unstripped.nii.gz' for mod in modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return np.stack(images)

    def image_DTI(i, dti_modalities: Output, _base):
        path = _base / f'NIfTI-files/images_DTI/{i}'
        if not path.exists():
            return None
        image_pathes = [path / f'{i}_DTI_{mod}.nii.gz' for mod in dti_modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return np.stack(images)

    def image_DSC(i, dsc_modalities: Output, _base):
        path = _base / f'NIfTI-files/images_DSC/{i}'
        if not path.exists():
            return None
        image_pathes = [path / (f'{i}_DSC_{mod}.nii.gz' if mod else f'{i}_DSC.nii.gz') for mod in dsc_modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return images

    @lru_cache(1)
    def _clinical_info(_base):
        return pd.read_csv(_base / 'UPENN-GBM_clinical_info_v1.0.csv')

    @lru_cache(1)
    def _acqusition_info(_base):
        return pd.read_csv(_base / 'UPENN-GBM_acquisition.csv')

    def clinical_info(i, _clinical_info) -> ClinicalInfo:
        row = _clinical_info[_clinical_info.ID == i]
        return ClinicalInfo(*row.iloc[0, 1:])

    def acqusition_info(i, _acqusition_info) -> AcquisitionInfo:
        row = _acqusition_info[_acqusition_info.ID == i]
        return AcquisitionInfo(*row.iloc[0, 1:])

    def subject_id(i):
        return i.split('_')[0]

    def affine(i):
        return np.array([[-1.0, 0.0, 0.0, -0.0], [0.0, -1.0, 0.0, 239.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def spacing(i):
        return (1, 1, 1)
