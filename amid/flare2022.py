import gzip
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
from connectome import Source, Transform, meta
from connectome.interface.nodes import Silent

from .internals import normalize


class FLARE2022Base(Source):
    """
    An abdominal organ segmentation dataset for semi-supervised learning [1]_.

    The dataset was used at the MICCAI FLARE 2022 challenge.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download link: https://flare22.grand-challenge.org/Dataset/

    The `root` folder should contain the two downloaded folders, namely: "Training" and "Validation".

    Examples
    --------
    >>> # Place the downloaded folders in any folder and pass the path to the constructor:
    >>> ds = FLARE2022(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 2100
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 110)
    >>> print(ds.mask(ds.ids[25]).shape)
    # (512, 512, 104)

    References
    ----------
    .. [1] Ma, Jun, et al. "Fast and Low-GPU-memory abdomen CT organ segmentation: The FLARE challenge."
    Medical Image Analysis 82 (2022): 102616.
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the locations of the downloaded data.')

        result = set()

        # 50 Training Labeled cases
        archive = Path(_root) / 'Training' / 'FLARE22_LabeledCase50' / 'images.zip'
        with ZipFile(archive) as zf:
            for file in zf.namelist():
                result.add(f"TL{file.split('_')[-2]}")

        # 2000 Training Unlabeled cases
        for archive in (Path(_root) / 'Training').glob('*.zip'):
            with ZipFile(archive) as zf:
                for file in zf.namelist():
                    if not file.endswith('.nii.gz'):
                        continue

                    file = Path(file)
                    result.add(f"TU{file.name.split('_')[-2]}")

        # 50 Validation Unlabeled cases
        for file in (Path(_root) / 'Validation').glob('*'):
            if not file.name.endswith('.nii.gz'):
                continue

            result.add(f"VU{file.name.split('_')[-2]}")

        return sorted(result)

    def _file(i, _root: Silent):
        # 50 Training Labeled cases
        if i.startswith('TL'):
            archive = Path(_root) / 'Training' / 'FLARE22_LabeledCase50' / 'images.zip'
            with ZipFile(archive) as zf:
                for file in zf.namelist():
                    if i[2:] in file:
                        return zipfile.Path(archive, file)

        # 2000 Training Unlabeled cases
        for archive in (Path(_root) / 'Training').glob('*.zip'):
            with ZipFile(archive) as zf:
                for file in zf.namelist():
                    if i[2:] in file:
                        return zipfile.Path(archive, file)

        # 50 Validation Unlabeled cases
        if i.startswith('VU'):
            file = Path(_root) / 'Validation' / f'FLARETs_{i[2:]}_0000.nii.gz'
            return file

        raise ValueError(f'Id "{i}" not found')

    def image(_file):
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.asarray(image.dataobj)

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation"""
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    def mask(i, _root: Silent):
        if not i.startswith('TL'):
            return None

        archive = Path(_root) / 'Training' / 'FLARE22_LabeledCase50' / 'labels.zip'
        with ZipFile(archive) as zf:
            for file in zf.namelist():
                if i[2:] in file:
                    with zipfile.Path(archive, file).open('rb') as opened:
                        with gzip.GzipFile(fileobj=opened) as nii:
                            nii = nibabel.FileHolder(fileobj=nii)
                            mask = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                            return np.asarray(mask.dataobj)


class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nibabel.affines.voxel_sizes(affine)


FLARE2022 = normalize(
    FLARE2022Base,
    'FLARE2022',
    'flare2022',
    body_region='Abdomen',
    license=None,
    link='https://flare22.grand-challenge.org/',
    modality='CT',
    prep_data_size='347G',
    raw_data_size='247G',
    task='Semi-supervised abdominal organ segmentation',
    normalizers=[SpacingFromAffine()],
)
