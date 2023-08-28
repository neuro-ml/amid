# TODO:
# 1. all todos
# 2. check every field for every `id`
# for i in ds.ids:
#     for field in [dir(ds)]:
#         ds.get(field, i)
# 3. run ./lint.sh


import contextlib
import tarfile
from pathlib import Path
import gzip

import nibabel as nb
import numpy as np
from connectome import Output, Source, meta, Transform
from connectome.interface.nodes import Silent

from .internals import checksum, licenses, register
from .utils import deprecate


@register(
    body_region='Abdominal',
    license=licenses.CC_BYSA_40,
    link='https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar',
    modality='CT',
    # prep_data_size='',
    raw_data_size='26.9G',
    task='Liver and tumour segmentation',
)

@checksum('msd')
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
    """

    _root: str = None

#TODO adopt for all datasets in MSD
    @meta
    def ids(_root: Silent):
        result = set()
        tasks = ("Task03_Liver.tar", ...)
        for task in tasks:

            with tarfile.open(Path(_root) / task) as tf:
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
    
    def train_test(i):
        if "train" in i: return "train"
        else: return "test"

#TODO from json
    def image_modality(i) -> str:
        None

# TODO
    def task(i) -> str:
        task_dict = {'liver':"Task03_Liver",
         'spleen':....}
        
        for key,value in task_dict.items():
            if key in i:
                return  value
            

#TODO from json
    def segmentation_labels(i) -> dict:
        """Returns segmentation labels for the task
        """
        return 

#TODO use self.task instead of hardcoded  'Task03_Liver'
    def _file(i, task, _root: Silent):
        num_id = i.split('_')[-1]
        if 'train' in i:
            return Path(_root) / task / 'imagesTr'/ f'liver_{num_id}.nii.gz'
        else:
            return Path(_root) / 'Task03_Liver'/ 'imagesTs'/ f'liver_{num_id}.nii.gz'

    def image(_file):
        with open_nii_gz_file(_file) as nii_image:
            # most CT/MRI scans are integer-valued, this will help us improve compression rates
            return np.int16(nii_image.get_fdata())
        
    def mask(_file):
        if 'Ts' not in str(_file.parent):
            with open_nii_gz_file(Path(str(_file).replace('images', 'labels'))) as nii_image:
                return np.uint8(nii_image.get_fdata())
        else:
            return None

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz_file(_file) as nii_image:
            return nii_image.affine

    
    @classmethod
    def normalizer(cls):
        return SpacingFromAffine()


@contextlib.contextmanager
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})


class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nb.affines.voxel_sizes(affine)