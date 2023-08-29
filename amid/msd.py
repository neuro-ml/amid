import contextlib
import tarfile
from pathlib import Path
import gzip
import os
import json

import nibabel as nb
import numpy as np
from connectome import Output, Source, meta, Transform
from connectome.interface.nodes import Silent

from .internals import checksum, licenses, register


@register(
    body_region=('Chest', 'Abdominal', 'Head'),
    license=licenses.CC_BYSA_40,
    link='https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2',
    modality=('CT', 'CE CT', 'MRI', 'MRI FLAIR', 'MRI T1w', 'MRI t1gd', 'MRI T2w', 'MRI T2', 'MRI ADC'),
    raw_data_size='97.8G',
    task='Medical Segmentation Decathlon',
)

@checksum('msd')
class MSD(Source):
    """
    MSD is a Medical Segmentaton Decathlon Challenge with 10 tasks.
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.
    Notes
    -----
    Data can be downloaded here: https://msd-for-monai.s3-us-west-2.amazonaws.com/
    or here: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2/
    Then, the folder with raw downloaded data should contain tar archive with data and masks
    (`Task03_Liver.tar`).
    """

    _root: str = None
    _task_to_name: dict = {
        "Task01_BrainTumour": "BRATS",
        "Task02_Heart": "heart",
        "Task03_Liver": "liver",
        "Task04_Hippocampus": "hippocampus",
        "Task05_Prostate": "prostate",
        "Task06_Lung": "lung",
        "Task07_Pancreas": "pancreas",
        "Task08_HepaticVessel": "hepaticvessel",
        "Task09_Spleen": "spleen",
        "Task10_Colon": "colon"
        }

    @meta
    def ids(_root: Silent) -> tuple: # _tasks):
        result = set()

        tar_files = [os.path.join(_root, f) for f in os.listdir(_root) if f.endswith('.tar')]
        for tar_file in tar_files:
            task = str(tar_file).split('/')[-1].split('.')[0]
            with tarfile.open(tar_file, 'r') as tf:
                for tarinfo in tf.getmembers():
                    if tarinfo.isdir():
                        continue
                    fold = '_train_'
                    if 'Ts' in tarinfo.path:
                        fold = '_test_'
                    file_stem = Path(tarinfo.path).stem
                    if file_stem.startswith('.') or not file_stem.endswith('.nii'):
                        continue

                    result.add(task + fold + file_stem.split('_')[1].split('.')[0])

        return tuple(sorted(result, key=lambda x: (x.split('_')[0], int(x.split('_')[-1]))))

    def train_test(i) -> str:
        if "train" in i: 
            fold = "train"
        else: 
            fold = "test"
        return fold

    def task(i) -> str:
        return "_".join(i.split("_")[:2])
    
    def _file(i, task: Output, _root: Silent, _task_to_name):
        name = _task_to_name[task]
        num_id = i.split('_')[-1]
        tar_path =  Path(_root) / f'{task}.tar'
        file_path = Path(task) / ('imagesTr' if 'train' in i else 'imagesTs') / f'{name}_{num_id}.nii.gz'
        return tar_path, str(file_path)

    def image(_file):
        tar_path, file_path = _file
        with open_nii_gz_from_tar(tar_path, file_path) as nii_image:
            # most CT/MRI scans are integer-valued, this will help us improve compression rates
            return np.int16(nii_image.get_fdata())
        
    def mask(_file):
        tar_path, file_path = _file
        if 'imagesTs' not in file_path:
            with open_nii_gz_from_tar(tar_path, file_path.replace('images', 'labels')) as nii_image:
                return np.uint8(nii_image.get_fdata())
        return

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation."""
        tar_path, file_path = _file
        with open_nii_gz_from_tar(tar_path, file_path) as nii_image:
            return nii_image.affine

    def image_modality(i, _root: Silent) -> str:
        task = "_".join(i.split("_")[:2])
        with tarfile.open(Path(_root) / f"{task}.tar") as tf:
            member = tf.getmember(f"{task}/dataset.json")
            file = tf.extractfile(member)
            return json.loads(file.read())["modality"]

    def segmentation_labels(i, _root: Silent) -> dict:
        """Returns segmentation labels for the task"""
        task = "_".join(i.split("_")[:2])
        with tarfile.open(Path(_root) / f"{task}.tar") as tf:
            member = tf.getmember(f"{task}/dataset.json")
            file = tf.extractfile(member)
            return json.loads(file.read())["labels"]

    @classmethod
    def normalizer(cls):
        return SpacingFromAffine()

@contextlib.contextmanager
def open_nii_gz_from_tar(tar_path, nii_gz_path):
    """Opens a .nii.gz file from inside a .tar archive.

    Parameters:
    - tar_path: path to the .tar archive.
    - nii_gz_path: path to the .nii.gz file inside the .tar archive.

    Yields:
    - nibabel.Nifti1Image object.
    """
    with tarfile.open(tar_path, 'r') as tar:
        with tar.extractfile(nii_gz_path) as extracted:
            with gzip.GzipFile(fileobj=extracted) as nii_gz:
                nii = nb.FileHolder(fileobj=nii_gz)
                yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})

class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nb.affines.voxel_sizes(affine)