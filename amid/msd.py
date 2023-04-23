import contextlib
import tarfile
from pathlib import Path

import nibabel as nb
import numpy as np
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent

from .internals import checksum, licenses, register
from .utils import deprecate


@register(
    body_region='Abdominal',
    # license=licenses.CC0_10,
    link='https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar',
    modality='CT',
    # prep_data_size='300M',
    # raw_data_size='310M',
    task='Liver and tumour segmentation',
)
@checksum('medseg9')
class Task03Liver(Source):
    """
    Task03Liver is a public liver and tumour  CT segmentation dataset from Medical Segmentaton Decathlon Challenge with 130 annotated images.
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.
    Notes
    -----
    Data can be downloaded here: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar.
    Then, the folder with raw downloaded data should contain tar archive with data and masks
    (`Task03_Liver.tar`).
    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = Medseg9(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 9
    >>> print(ds.image(ds.ids[0]).shape)
    # (630, 630, 45)
    >>> print(ds.covid(ds.ids[0]).shape)
    # (630, 630, 45)
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        result = set()

        with tarfile.open(Path(_root) / 'Task03_Liver.tar') as tf:
            for tarinfo in tf.getmembers():
                if tarinfo.isdir():
                    continue
                fold = 'train_'
                if 'Ts' in tarinfo.path:
                    fold = 'test_'
                file_stem = Path(tarinfo.path).stem
                if file_stem.startswith('.') or not file_stem.endswith('.nii'):
                    continue
                result.add(fold + file_stem.split('.nii')[0])

        return tuple(sorted(result))

    def _file(i, _root: Silent):
        num_id = i.split('_')[-1]
        if 'train' in i:
            return Path(_root) / 'Task03_Liver'/ 'imagesTr'/ f'liver_{num_id}.nii.gz'
        else:
            return Path(_root) / 'Task03_Liver'/ 'imagesTs'/ f'liver_{num_id}.nii.gz'

    def image(_file):
        with open_nii_gz_file(_file) as nii_image:
            # most CT/MRI scans are integer-valued, this will help us improve compression rates
            return np.int16(nii_image.get_fdata())
        
    def mask(_file):
        if 'Ts' not in _file.parent:
            with open_nii_gz_file(_file.parent.replace('images', 'labels') / _file.name) as nii_image:
                return np.uint8(nii_image.get_fdata())

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz_file(_file) as nii_image:
            return nii_image.affine

    @deprecate(message='Use spacing method instead.')
    def voxel_spacing(spacing: Output):
        return spacing

    def spacing(_file):
        """Returns voxel spacing along axes (x, y, z)."""
        with open_nii_gz_file(_file) as nii_image:
            return tuple(nii_image.header['pixdim'][1:4])


@contextlib.contextmanager
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})