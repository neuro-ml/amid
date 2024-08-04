from functools import cached_property

import nibabel as nb
import numpy as np
import pandas as pd

from ..internals import Dataset, licenses, register
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
class UPENN_GBM(Dataset):
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

    @property
    def ids(self):
        ids = [x.name for x in (self.root / 'NIfTI-files/images_structural').iterdir()]
        return tuple(sorted(ids))

    @property
    def modalities(self):
        return ['T1', 'T1GD', 'T2', 'FLAIR']

    @property
    def dsc_modalities(self):
        return ['', 'ap-rCBV', 'PH', 'PSR']

    @property
    def dti_modalities(self):
        return ['AD', 'FA', 'RD', 'TR']

    def _mask_path(self, i):
        p1 = self.root / 'NIfTI-files/images_segm'
        p2 = self.root / 'NIfTI-files/automated_segm'
        p1 = list(p1.glob(i + '*'))
        p2 = list(p2.glob(i + '*'))
        return p1[0] if p1 else p2[0] if p2 else None

    def mask(self, i):
        path = self._mask_path(i)
        if not path:
            return None
        return np.asarray(nb.load(path).get_fdata())

    def is_mask_automated(self, i):
        path = self._mask_path(i)
        if path is None:
            return None
        return path.parent.name == 'automated_segm'

    def image(self, i):
        path = self.root / f'NIfTI-files/images_structural/{i}'
        image_pathes = [path / f'{i}_{mod}.nii.gz' for mod in self.modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return np.stack(images)

    def image_unstripped(self, i):
        path = self.root / f'NIfTI-files/images_structural_unstripped/{i}'
        image_pathes = [path / f'{i}_{mod}_unstripped.nii.gz' for mod in self.modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return np.stack(images)

    def image_DTI(self, i):
        path = self.root / f'NIfTI-files/images_DTI/{i}'
        if not path.exists():
            return None
        image_pathes = [path / f'{i}_DTI_{mod}.nii.gz' for mod in self.dti_modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return np.stack(images)

    def image_DSC(self, i):
        path = self.root / f'NIfTI-files/images_DSC/{i}'
        if not path.exists():
            return None
        image_pathes = [path / (f'{i}_DSC_{mod}.nii.gz' if mod else f'{i}_DSC.nii.gz') for mod in self.dsc_modalities]
        images = [np.asarray(nb.load(p).dataobj) for p in image_pathes]
        return images

    @cached_property
    def _clinical_info(self):
        return pd.read_csv(self.root / 'UPENN-GBM_clinical_info_v1.0.csv')

    @cached_property
    def _acqusition_info(self):
        return pd.read_csv(self.root / 'UPENN-GBM_acquisition.csv')

    def clinical_info(self, i):
        row = self._clinical_info[self._clinical_info.ID == i]
        return ClinicalInfo(*row.iloc[0, 1:])

    def acqusition_info(self, i):
        row = self._acqusition_info[self._acqusition_info.ID == i]
        return AcquisitionInfo(*row.iloc[0, 1:])

    def subject_id(self, i):
        return i.split('_')[0]

    def affine(self, i):
        return np.array([[-1.0, 0.0, 0.0, -0.0], [0.0, -1.0, 0.0, 239.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def spacing(self, i):
        return (1, 1, 1)
