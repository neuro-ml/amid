import nibabel as nb
import numpy as np
import pandas as pd

from .internals import Dataset, field, licenses, register


@register(
    body_region='Head',
    license=licenses.PhysioNet_RHD_150,
    link='https://physionet.org/content/ct-ich/1.3.1/',
    modality='CT',
    prep_data_size='661M',
    raw_data_size='2,8G',
    task='Intracranial hemorrhage segmentation',
)
class CT_ICH(Dataset):
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

    @property
    def ids(self):
        result = [f'ct_ich_{uid:0=3d}' for uid in [*range(49, 59), *range(66, 131)]]
        return tuple(sorted(result))

    def _image_file(self, i):
        num_id = i.split('_')[-1]
        return nb.load(self.root / 'ct_scans' / f'{num_id}.nii')

    @field
    def image(self, i) -> np.ndarray:
        # most CT/MRI scans are integer-valued, this will help us improve compression rates
        return np.int16(self._image_file(i).get_fdata())

    @field
    def mask(self, i) -> np.ndarray:
        num_id = i.split('_')[-1]
        mask_path = self.root / 'masks' / f'{num_id}.nii'
        ct_scan_nifti = nb.load(mask_path)
        return ct_scan_nifti.get_fdata().astype(bool)

    @field
    def affine(self, i) -> np.ndarray:
        """The 4x4 matrix that gives the image's spatial orientation."""
        return self._image_file(i).affine

    def spacing(self, i):
        """Returns voxel spacing along axes (x, y, z)."""
        return tuple(self._image_file(i).header['pixdim'][1:4])

    @property
    def _patient_metadata(self):
        return pd.read_csv(self.root / 'Patient_demographics.csv', index_col='Patient Number')

    @property
    def _diagnosis_metadata(self):
        return pd.read_csv(self.root / 'hemorrhage_diagnosis_raw_ct.csv')

    def _row(self, i):
        patient_id = int(i.split('_')[-1])
        return self._patient_metadata.loc[patient_id]

    @field
    def age(self, i) -> float:
        return self._row(i)['Age\n(years)']

    @field
    def sex(self, i) -> str:
        return self._row(i)['Gender']

    @field
    def intraventricular_hemorrhage(self, i) -> bool:
        """Returns True if hemorrhage exists and its type is intraventricular."""
        num_id = int(i.split('_')[-1])
        return str(self._patient_metadata['Hemorrhage type based on the radiologists diagnosis '].loc[num_id]) != 'nan'

    @field
    def intraparenchymal_hemorrhage(self, i) -> bool:
        """Returns True if hemorrhage was diagnosed and its type is intraparenchymal."""
        num_id = int(i.split('_')[-1])
        return str(self._patient_metadata['Unnamed: 4'].loc[num_id]) != 'nan'

    @field
    def subarachnoid_hemorrhage(self, i) -> bool:
        """Returns True if hemorrhage was diagnosed and its type is subarachnoid."""
        num_id = int(i.split('_')[-1])
        return str(self._patient_metadata['Unnamed: 5'].loc[num_id]) != 'nan'

    @field
    def epidural_hemorrhage(self, i) -> bool:
        """Returns True if hemorrhage was diagnosed and its type is epidural."""
        num_id = int(i.split('_')[-1])
        return str(self._patient_metadata['Unnamed: 6'].loc[num_id]) != 'nan'

    @field
    def subdural_hemorrhage(self, i) -> bool:
        """Returns True if hemorrhage was diagnosed and its type is subdural."""
        num_id = int(i.split('_')[-1])
        return str(self._patient_metadata['Unnamed: 7'].loc[num_id]) != 'nan'

    @field
    def fracture(self, i) -> bool:
        """Returns True if skull fracture was diagnosed."""
        num_id = int(i.split('_')[-1])
        return str(self._patient_metadata['Fracture (yes 1/no 0)'].loc[num_id]) != 'nan'

    @field
    def notes(self, i) -> str:
        """Returns special notes if they exist."""
        num_id = int(i.split('_')[-1])
        result = str(self._patient_metadata['Note1'].loc[num_id])
        return result if result != 'nan' else None

    @field
    def hemorrhage_diagnosis_raw_metadata(self, i):
        num_id = int(i.split('_')[-1])
        return self._diagnosis_metadata[self._diagnosis_metadata['PatientNumber'] == num_id]
