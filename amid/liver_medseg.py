import re
import zipfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from amid.cc359 import open_nii_gz_file
from amid.internals import checksum, register


@register(
    body_region=('Chest', 'Abdomen'),
    license='CC-BY-SA 4.0',
    link='https://www.medseg.ai/database/liver-segments-50-cases',
    modality='CT',
    prep_data_size=None,  # TODO: should be measured...
    raw_data_size='616M',
    task='Segmentation',
)
@checksum('liver_medseg')
class LiverMedseg(Source):
    """
    LiverMedseg is a public CT segmentation dataset with 50 annotated images.
    Case collection of 50 livers with their segments.
    Images obtained from Decathlon Medical Segmentation competition

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Download links:
    https://www.medseg.ai/database/liver-segments-50-cases

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = LiverMedseg(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 50
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 38)

    References
    ----------
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        result = set()

        with ZipFile(Path(_root) / 'img.zip') as zf:
            for zipinfo in zf.infolist():

                if zipinfo.is_dir():
                    continue
                file_stem = Path(zipinfo.filename).stem
                result.add('liver_medseg_' + re.findall(r'\d+', file_stem)[0])

        return tuple(sorted(result))

    def _file(i: str, _root: Silent):
        num_id = i.split('_')[-1]
        return zipfile.Path(Path(_root) / 'img.zip', f'img{num_id}.nii.gz')

    def image(_file) -> np.ndarray:
        with open_nii_gz_file(_file) as nii_file:
            return np.asarray(nii_file.dataobj)

    def affine(_file) -> np.ndarray:
        """The 4x4 matrix that gives the image's spatial orientation."""
        with open_nii_gz_file(_file) as nii_file:
            return nii_file.affine

    def voxel_spacing(_file) -> tuple:
        with open_nii_gz_file(_file) as nii_file:
            return tuple(nii_file.header['pixdim'][1:4])

    def mask(_file) -> np.ndarray:
        path = Path(str(_file).replace('img', 'mask'))
        folder, image = path.parent, path.name
        _file = zipfile.Path(folder, image)
        with open_nii_gz_file(_file) as nii_file:
            return np.asarray(nii_file.dataobj).astype(np.uint8)
