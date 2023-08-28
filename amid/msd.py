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
    # There is a link to AWS, I don't know which is better
    link='https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2',
    modality=('CT', 'MRI', 'MRI FLAIR', 'MRI T1w', 'MRI t1gd', 'MRI T2w', 'MRI T2', 'MRI ADC'),
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
    _task_to_name: dict = {"Task01_BrainTumour": "BRATS",
                            "Task02_Heart": "heart",
                            "Task03_Liver": "liver",
                            "Task04_Hippocampus": "hippocampus",
                            "Task05_Prostate": "prostate",
                            "Task06_Lung": "lung",
                            "Task07_Pancreas": "pancreas",
                            "Task08_HepaticVessel": "hepaticvessel",
                            "Task09_Spleen": "spleen",
                            "Task10_Colon": "colon",
                            }

    @meta
    def ids(_root: Silent) -> tuple: # _tasks):
        result = set()

        tar_files = [os.path.join(_root, f) for f in os.listdir(_root) if f.endswith('.tar')]
        for tar_file in tar_files:
            task = str(tar_file).split('/')[-1].split('.')[0]
            with tarfile.open(tar_file, 'r') as tf:

        # for task in _tasks.keys():
        #     with tarfile.open(Path(_root) / f"{task}.tar") as tf:

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
        if "train" in i: return "train"
        else: return "test"

    def task(i) -> str:
        return "_".join(i.split("_")[:2])
    
    # def _file(i, task, _root: Silent, _task_to_name): swears at it
    def _file(i, _root: Silent, _task_to_name):
        task = "_".join(i.split("_")[:2])
        name = _task_to_name[task]
        num_id = i.split('_')[-1]
        return Path(_root) / task / ('imagesTr' if 'train' in i else 'imagesTs') / f'{name}_{num_id}.nii.gz'

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
def open_nii_gz_file(file):
    with file.open('rb') as opened:
        with gzip.GzipFile(fileobj=opened) as nii:
            nii = nb.FileHolder(fileobj=nii)
            yield nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})


class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nb.affines.voxel_sizes(affine)