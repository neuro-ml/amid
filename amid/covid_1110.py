import gzip
from pathlib import Path

import nibabel
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from .internals import checksum, register


@register(
    body_region='Thorax',
    modality='CT',
    task='COVID-19 Segmentation',
    link='https://mosmed.ai/en/datasets/covid191110/',
    raw_data_size='21G',
)
@checksum('covid_1110')
class MoscowCovid1110(Source):
    """
    The Moscow Radiology COVID-19 dataset.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded files.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Download links:
    https://mosmed.ai/en/datasets/covid191110/

    Examples
    --------
    >>> # Place the downloaded files in any folder and pass the path to the constructor:
    >>> ds = MoscowCovid1110(root='/path/to/files/root')
    >>> print(len(ds.ids))
    # 1110
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 43)
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the locations of the zip archives')

        return sorted({f.name[:-7] for f in Path(_root).glob('CT-*/*')})

    def _file(i, _root: Silent):
        return next(Path(_root).glob(f'CT-*/{i}.nii.gz'))

    def image(_file):
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                # most ct scans are integer-valued, this will help us improve compression rates
                #  (instead of using `image.get_fdata()`)
                return np.asarray(image.dataobj)

    def affine(_file):
        with _file.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    def label(_file):
        return _file.parent.name[3:]

    def mask(i, _root: Silent):
        path = Path(_root) / 'masks' / f'{i}_mask.nii.gz'
        if not path.exists():
            return

        with path.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nibabel.FileHolder(fileobj=nii)
                image = nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.asarray(image.dataobj) > 0.5
