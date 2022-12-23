import contextlib
import gzip
import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel
import numpy as np
import pandas as pd
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import checksum, register
from .anatomical_structures import ANATOMICAL_STRUCTURES


@register(
    body_region=('Head', 'Thorax', 'Abdomen', 'Pelvis', 'Legs'),
    license='CC BY 4.0',
    link='https://zenodo.org/record/6802614#.Y6M2MxXP1D8',
    modality='CT',
    raw_data_size='35G',
    prep_data_size='35G',
    task='Supervised anatomical structures segmentation',
)
@checksum('totalsegmentator')
class Totalsegmentator(Source):
    """
    In 1204 CT images we segmented 104 anatomical structures (27 organs, 59 bones, 10 muscles, 8 vessels)
    covering a majority of relevant classes for most use cases.

    The CT images were randomly sampled from clinical routine, thus representing a real world dataset which
    generalizes to clinical application.

    The dataset contains a wide range of different pathologies, scanners, sequences and institutions. [1]

    Parameters
    ----------
    root : str, Path, optional
        absolute path to the downloaded archive.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download link: https://zenodo.org/record/6802614/files/Totalsegmentator_dataset.zip

    Examples
    --------
    >>> # Download the archive to any folder and pass the path to the constructor:
    >>> ds = Totalsegmentator(root='/path/to/the/downloaded/archive')
    >>> print(len(ds.ids))
    # 1203
    >>> print(ds.image(ds.ids[0]).shape)
    # (294, 192, 179)
    >>> print(ds.aorta(ds.ids[25]).shape)
    # (320, 320, 145)

    References
    ----------
    .. [1] Jakob Wasserthal (2022) Dataset with segmentations of 104 important anatomical structures in 1204 CT images.
    Available at: https://zenodo.org/record/6802614#.Y6M2MxXP1D8
    """

    _root: str = None

    @staticmethod
    def add_masks(scope):
        def make_loader(anatomical_structure):
            def loader(i, _root: Silent):
                file = f'Totalsegmentator_dataset/{i}/segmentations/{anatomical_structure}.nii.gz'

                with unpack(_root, file) as (unpacked, is_unpacked):
                    if is_unpacked:
                        return np.asarray(nibabel.load(file).dataobj)
                    else:
                        with gzip.GzipFile(fileobj=unpacked) as nii:
                            nii = nibabel.FileHolder(fileobj=nii)
                            image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})

                            return np.asarray(image.dataobj)

            return loader

        for anatomical_structure in ANATOMICAL_STRUCTURES:
            scope[anatomical_structure] = make_loader(anatomical_structure)

    add_masks(locals())

    @meta
    def ids(_root: Silent):
        with ZipFile(_root) as zf:
            namelist = [x.rstrip('/') for x in zf.namelist()]

            ids = []
            for f in namelist:
                if len(f.split('/')) == 2 and f.split('/')[-1] != 'meta.csv':
                    ids.append(f.split('/')[-1])

            return sorted(ids)

    def meta(_root: Silent):
        file = 'Totalsegmentator_dataset/meta.csv'

        with ZipFile(_root) as zf:
            return pd.read_csv(zf.open(file), sep=';').head()

    def image(i, _root: Silent):
        file = f'Totalsegmentator_dataset/{i}/ct.nii.gz'

        with unpack(_root, file) as (unpacked, is_unpacked):
            if is_unpacked:
                return np.asarray(nibabel.load(file).dataobj)
            else:
                with gzip.GzipFile(fileobj=unpacked) as nii:
                    nii = nibabel.FileHolder(fileobj=nii)
                    image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})

                    return np.asarray(image.dataobj)

    def affine(i, _root: Silent):
        """The 4x4 matrix that gives the image's spatial orientation"""
        file = f'Totalsegmentator_dataset/{i}/ct.nii.gz'

        with unpack(_root, file) as (unpacked, is_unpacked):
            if is_unpacked:
                return nibabel.load(unpacked).affine
            else:
                with gzip.GzipFile(fileobj=unpacked) as nii:
                    nii = nibabel.FileHolder(fileobj=nii)
                    image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})

                    return image.affine


@contextlib.contextmanager
def unpack(root: str, relative: str):
    unpacked = Path(root) / relative
    if unpacked.exists():
        yield unpacked, True
    else:
        with zipfile.Path(root, relative).open('rb') as unpacked:
            yield unpacked, False
