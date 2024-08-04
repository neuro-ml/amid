import contextlib
import gzip
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel as nb
import numpy as np

from ..internals import Dataset, licenses, register


@register(
    body_region='Head',
    license=licenses.CC_BYND_40,
    link='https://sites.google.com/view/calgary-campinas-dataset/home',
    modality='MRI T1',
    prep_data_size='14,66G',
    raw_data_size='4,1G',
    task='Segmentation',
)
class CC359(Dataset):
    """
    A (C)algary-(C)ampinas public brain MR dataset with (359) volumetric images [1]_.

    There are three segmentation tasks on this dataset: (i) brain, (ii) hippocampus, and
    (iii) White-Matter (WM), Gray-Matter (WM), and Cerebrospinal Fluid (CSF) segmentation.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.


    Notes
    -----
    homepage (upd): https://sites.google.com/view/calgary-campinas-dataset/home
    homepage (old): https://miclab.fee.unicamp.br/calgary-campinas-359-updated-05092017

    To obtain MR images and brain and hippocampus segmentation masks, please, follow the instructions
    at the download platform: https://portal.conp.ca/dataset?id=projects/calgary-campinas.

    Via `datalad` lib you need to download three zip archives:
        - `Original.zip` (the original MR images)
        - `hippocampus_staple.zip` (Silver-standard hippocampus masks generated using STAPLE)
        - `Silver-standard-machine-learning.zip` (Silver-standard brain masks generated using a machine learning method)

    To the current date, WM, GM, and CSF mask could be downloaded only from the google drive:
    https://drive.google.com/drive/u/0/folders/0BxLb0NB2MjVZNm9JY1pWNFp6WTA?resourcekey=0-2sXMr8q-n2Nn6iY3PbBAdA.

    Here you need to manually download a folder (from the google drive root above)
    `CC359/Reconstructed/CC359/WM-GM-CSF/`

    So the root folder to pass to this dataset class should contain four objects:
        - three zip archives (`Original.zip`, `hippocampus_staple.zip`, and `Silver-standard-machine-learning.zip`)
        - one folder `WM-GM-CSF` with the original structure:
            <...>/WM-GM-CSF/CC0319_ge_3_45_M.nii.gz
            <...>/WM-GM-CSF/CC0324_ge_3_56_M.nii.gz
            ...

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> cc359 = CC359(root='/path/to/downloaded/data/folder/')
    >>> print(len(cc359.ids))
    # 359
    >>> print(cc359.image(cc359.ids[0]).shape)
    # (171, 256, 256)
    >>> print(cc359.wm_gm_csf(cc359.ids[80]).shape)
    # (180, 240, 240)

    References
    ----------
    .. [1] Souza, Roberto, et al. "An open, multi-vendor, multi-field-strength brain MR dataset
           and analysis of publicly available skull stripping methods agreement."
           NeuroImage 170 (2018): 482-494.
           https://www.sciencedirect.com/science/article/pii/S1053811917306687

    """

    @property
    def ids(self):
        result = set()
        with ZipFile(self.root / 'Original.zip') as zf:
            for zipinfo in zf.infolist():
                if zipinfo.is_dir():
                    continue

                file_name = Path(zipinfo.filename).name
                if file_name.startswith('CC'):
                    result.add(file_name.split('_')[0])

        return tuple(sorted(result))

    def _image_file(self, i):
        return get_zipfile(i, 'Original.zip', self.root)

    def vendor(self, i):
        return zipfile2meta(self._image_file(i))['vendor']

    def field(self, i):
        return zipfile2meta(self._image_file(i))['field']

    def age(self, i):
        return zipfile2meta(self._image_file(i))['age']

    def sex(self, i):
        return zipfile2meta(self._image_file(i))['gender']

    def image(self, i):
        with open_nii_gz_file(self._image_file(i)) as nii_image:
            return np.asarray(nii_image.dataobj)

    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz_file(self._image_file(i)) as nii_image:
            return nii_image.affine

    def spacing(self, i):
        """Returns voxel spacing along axes (x, y, z)."""
        with open_nii_gz_file(self._image_file(i)) as nii_image:
            return tuple(nii_image.header['pixdim'][1:4])

    # masks:

    def brain(self, i):
        zf = get_zipfile(i, 'Silver-standard-machine-learning.zip', self.root)
        with open_nii_gz_file(zf) as nii_image:
            return np.uint8(nii_image.get_fdata())

    def hippocampus(self, i):
        try:
            zf = get_zipfile(i, 'hippocampus_staple.zip', self.root)
        except KeyError:
            return None

        with open_nii_gz_file(zf) as nii_image:
            return np.uint8(nii_image.get_fdata())

    def wm_gm_csf(self, i):
        for file in (self.root / 'WM-GM-CSF').glob('*'):
            if file.name.startswith(i):
                with open_nii_gz_file(file) as nii_image:
                    return np.uint8(nii_image.get_fdata())


# TODO: sync with amid.utils
@contextlib.contextmanager
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})


def get_zipfile(_id, archive_name, root):
    archive = Path(root) / archive_name
    with ZipFile(archive) as zf:
        for zipinfo in zf.infolist():
            if Path(zipinfo.filename).name.startswith(_id) and not zipinfo.is_dir():
                return zipfile.Path(str(archive), zipinfo.filename)

    raise KeyError(f'Id "{_id}" not found')


def zipfile2meta(zf):
    return dict(zip(['id', 'vendor', 'field', 'age', 'gender'], zf.name[: -len('.nii.gz')].split('_')))
