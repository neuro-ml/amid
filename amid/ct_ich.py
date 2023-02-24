from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent

from .internals import checksum, licenses, register
from .utils import deprecate


@register(
    body_region='Head',
    license=licenses.PhysioNet_RHD_150,
    link='https://physionet.org/content/ct-ich/1.3.1/',
    modality='CT',
    prep_data_size='661M',
    raw_data_size='2,8G',
    task='Intracranial hemorrhage segmentation',
)
@checksum('ct_ich')
class CT_ICH(Source):
    """
    (C)omputed (T)omography Images for (I)ntracranial (H)emorrhage Detection and (S)egmentation.

    This dataset contains 75 head CT scans including 36 scans for patients diagnosed with
    intracranial hemorrhage with the following types:
    Intraventricular, Intraparenchymal, Subarachnoid, Epidural and Subdural.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Data can be downloaded here: https://physionet.org/content/ct-ich/1.3.1/.
    Then, the folder with raw downloaded data should contain folders `ct_scans` and `masks` along with other files.

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = CT_ICH(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 75
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 39)
    >>> print(ds.mask(ds.ids[0]).shape)
    # (512, 512, 39)
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        result = [f'ct_ich_{uid:0=3d}' for uid in np.concatenate([range(49, 59), range(66, 131)])]
        return tuple(sorted(result))

    def _image_file(i, _root: Silent):
        num_id = i.split('_')[-1]
        return nb.load(Path(_root) / 'ct_scans' / f'{num_id}.nii')

    def image(_image_file):
        # most CT/MRI scans are integer-valued, this will help us improve compression rates
        return np.int16(_image_file.get_fdata()[...])

    def mask(i, _root: Silent):
        num_id = i.split('_')[-1]
        mask_path = Path(_root) / 'masks' / f'{num_id}.nii'
        ct_scan_nifti = nb.load(mask_path)
        return np.bool_(ct_scan_nifti.get_fdata()[...])

    def affine(_image_file):
        """The 4x4 matrix that gives the image's spatial orientation."""
        return _image_file.affine

    @deprecate(message='Use `spacing` method instead.')
    def voxel_spacing(spacing: Output):
        return spacing

    def spacing(_image_file):
        """Returns voxel spacing along axes (x, y, z)."""
        return tuple(_image_file.header['pixdim'][1:4])

    def _patient_metadata(_root: Silent):
        return pd.read_csv(Path(_root) / 'Patient_demographics.csv', index_col='Patient Number')

    def _diagnosis_metadata(_root: Silent):
        return pd.read_csv(Path(_root) / 'hemorrhage_diagnosis_raw_ct.csv')

    def age(i, _patient_metadata):
        num_id = int(i.split('_')[-1])
        _patient_metadata['Age\n(years)'].loc[num_id]

    def gender(i, _patient_metadata):
        num_id = int(i.split('_')[-1])
        _patient_metadata['Gender'].loc[num_id]

    def intraventricular_hemorrhage(i, _patient_metadata):
        """Returns True if hemorrhage exists and its type is intraventricular."""
        num_id = int(i.split('_')[-1])
        return str(_patient_metadata['Hemorrhage type based on the radiologists diagnosis '].loc[num_id]) != 'nan'

    def intraparenchymal_hemorrhage(i, _patient_metadata):
        """Returns True if hemorrhage was diagnosed and its type is intraparenchymal."""
        num_id = int(i.split('_')[-1])
        return str(_patient_metadata['Unnamed: 4'].loc[num_id]) != 'nan'

    def subarachnoid_hemorrhage(i, _patient_metadata):
        """Returns True if hemorrhage was diagnosed and its type is subarachnoid."""
        num_id = int(i.split('_')[-1])
        return str(_patient_metadata['Unnamed: 5'].loc[num_id]) != 'nan'

    def epidural_hemorrhage(i, _patient_metadata):
        """Returns True if hemorrhage was diagnosed and its type is epidural."""
        num_id = int(i.split('_')[-1])
        return str(_patient_metadata['Unnamed: 6'].loc[num_id]) != 'nan'

    def subdural_hemorrhage(i, _patient_metadata):
        """Returns True if hemorrhage was diagnosed and its type is subdural."""
        num_id = int(i.split('_')[-1])
        return str(_patient_metadata['Unnamed: 7'].loc[num_id]) != 'nan'

    def fracture(i, _patient_metadata):
        """Returns True if skull fracture was diagnosed."""
        num_id = int(i.split('_')[-1])
        return str(_patient_metadata['Fracture (yes 1/no 0)'].loc[num_id]) != 'nan'

    def notes(i, _patient_metadata):
        """Returns special notes if they exist."""
        num_id = int(i.split('_')[-1])
        result = str(_patient_metadata['Note1'].loc[num_id])
        return result if result != 'nan' else None

    def hemorrhage_diagnosis_raw_metadata(i, _diagnosis_metadata):
        num_id = int(i.split('_')[-1])
        return _diagnosis_metadata[_diagnosis_metadata['PatientNumber'] == num_id]
