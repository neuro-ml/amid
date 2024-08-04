import contextlib
import gzip
import json
import tarfile
from pathlib import Path

import nibabel as nb
import numpy as np

from .internals import Dataset, register


@register(
    body_region=('Chest', 'Abdominal', 'Head'),
    link='http://medicaldecathlon.com/',
    modality=('CT', 'CE CT', 'MRI', 'MRI FLAIR', 'MRI T1w', 'MRI t1gd', 'MRI T2w', 'MRI T2', 'MRI ADC'),
    raw_data_size='97.8G',
    task='Image segmentation',
)
class MSD(Dataset):
    """
    MSD is a Medical Segmentaton Decathlon Challenge with 10 tasks.
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Data can be downloaded here:http://medicaldecathlon.com/
    or here: https://msd-for-monai.s3-us-west-2.amazonaws.com/
    or here: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2/
    Then, the folder with raw downloaded data should contain tar archive with data and masks
    (`Task03_Liver.tar`).
    """

    @property
    def ids(self):
        ids_all = []
        for folder in self.root.glob('*'):
            if folder.name.endswith('.tar'):
                ids_folder = ids_from_tar(folder)
            else:
                ids_folder = ids_from_folder(folder)
            ids_all.extend(ids_folder)
        return tuple(ids_all)

    def train_test(self, i) -> str:
        fold = 'train' if 'train' in i else 'test'
        return fold

    def task(self, i) -> str:
        return NAME_TO_TASK[i.split('_')[1]]

    def _relative(self, i):
        name = i.removeprefix('train_').removeprefix('test_')
        return Path(self.task(i)), Path('imagesTr' if 'train' in i else 'imagesTs') / f'{name}.nii.gz'

    def image(self, i):
        with open_nii_gz(self.root, self._relative(i)) as (file, unpacked):
            if unpacked:
                return np.int16(nb.load(file).get_fdata())
            else:
                with gzip.GzipFile(fileobj=file) as nii_gz:
                    nii = nb.FileHolder(fileobj=nii_gz)
                    return np.int16(nb.Nifti1Image.from_file_map({'header': nii, 'image': nii}).get_fdata())

    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz(self.root, self._relative(i)) as (file, unpacked):
            if unpacked:
                return nb.load(file).affine
            else:
                with gzip.GzipFile(fileobj=file) as nii_gz:
                    nii = nb.FileHolder(fileobj=nii_gz)
                    return nb.Nifti1Image.from_file_map({'header': nii, 'image': nii}).affine

    def image_modality(self, i):
        task = self.task(i)
        if (self.root / task).is_dir():
            with open(self.root / task / 'dataset.json', 'r') as file:
                return json.loads(file.read())['modality']

        with tarfile.open(self.root / f'{task}.tar') as tf:
            member = tf.getmember(f'{task}/dataset.json')
            file = tf.extractfile(member)
            return json.loads(file.read())['modality']

    def segmentation_labels(self, i):
        """Returns segmentation labels for the task"""
        task = self.task(i)
        if (self.root / task).is_dir():
            with open(self.root / task / 'dataset.json', 'r') as file:
                return json.loads(file.read())['labels']

        with tarfile.open(self.root / f'{task}.tar') as tf:
            member = tf.getmember(f'{task}/dataset.json')
            file = tf.extractfile(member)
            return json.loads(file.read())['labels']

    def mask(self, i):
        task, relative = self._relative(i)
        if 'imagesTs' not in str(relative):
            with open_nii_gz(self.root, (task, str(relative).replace('images', 'labels'))) as (file, unpacked):
                if unpacked:
                    return np.uint8(nb.load(file).get_fdata())
                else:
                    with gzip.GzipFile(fileobj=file) as nii_gz:
                        nii = nb.FileHolder(fileobj=nii_gz)
                        return np.uint8(nb.Nifti1Image.from_file_map({'header': nii, 'image': nii}).get_fdata())


TASK_TO_NAME: dict = {
    'Task01_BrainTumour': 'BRATS',
    'Task02_Heart': 'la',
    'Task03_Liver': 'liver',
    'Task04_Hippocampus': 'hippocampus',
    'Task05_Prostate': 'prostate',
    'Task06_Lung': 'lung',
    'Task07_Pancreas': 'pancreas',
    'Task08_HepaticVessel': 'hepaticvessel',
    'Task09_Spleen': 'spleen',
    'Task10_Colon': 'colon',
}

NAME_TO_TASK = dict(zip(TASK_TO_NAME.values(), TASK_TO_NAME.keys()))


@contextlib.contextmanager
def open_nii_gz(path, nii_gz_path):
    """Opens a .nii.gz file from inside a .tar archive.

    Parameters:
    - path: path to the .tar archive or folder
    - nii_gz_path: path to the .nii.gz file inside the .tar archive.

    Yields:
    - nibabel.Nifti1Image object.
    """
    task, relative = nii_gz_path
    if (path / task / relative).exists():
        yield path / task / relative, True
    else:
        with tarfile.open(path / f'{task}.tar', 'r') as tar:
            yield tar.extractfile(str(task / relative)), False


def get_id(filename: Path):
    fold = 'test' if 'imagesTs' in str(filename) else 'train'
    name = filename.name.removesuffix('.nii.gz')
    return '_'.join([fold, name])


def ids_from_tar(tar_folder):
    ids = []
    with tarfile.open(tar_folder, 'r') as tf:
        for file in tf.getmembers():
            filename = Path(file.name)
            if not filename.name.startswith('._') and filename.suffix == '.gz' and 'images' in filename.parent.name:
                ids.append(get_id(filename))
    return sorted(ids)


def ids_from_folder(folder):
    ids = []
    for filename in folder.rglob('*.nii.gz'):
        if not filename.name.startswith('._') and filename.suffix == '.gz' and 'images' in filename.parent.name:
            ids.append(get_id(filename))
    return sorted(ids)
