import zipfile
from pathlib import Path
from typing import Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
from connectome import Source, meta
from connectome.interface.nodes import Output, Silent

from .internals import checksum, licenses, register
from .utils import open_nii_gz_file, unpack


@register(
    body_region='Head',
    license=licenses.CC_BYNCSA_40,
    link='http://www.braintumorsegmentation.org/',
    modality=('MRI T1', 'MRI T1Gd', 'MRI T2', 'MRI T2-FLAIR'),
    prep_data_size='8,96G',
    raw_data_size='15G',
    task=('Segmentation', 'Classification', 'Domain Adaptation'),
)
@checksum('brats2021')
class BraTS2021(Source):
    """
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
    2021: http://www.braintumorsegmentation.org/

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = BraTS2021(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 5880
    >>> print(ds.image(ds.ids[0]).shape)
    # (240, 240, 155)

    References
    ----------
    """

    _root: str = None

    def _base(_root: Silent) -> Path:
        if _root is None:
            raise ValueError('Please pass the locations of the zip archives')
        return Path(_root)

    @meta
    def ids(_base):
        result = set()
        for archive in _base.glob('*.zip'):
            if "TrainingData" not in str(archive) and "ValidationData" not in str(archive):
                continue
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue

                    file = Path(zipinfo.filename)
                    assert file.stem not in result, file.stem

                    if file.suffix == ".gz" and "seg" not in file.name:
                        result.add(file.stem.replace(".nii", ""))
                    else:
                        continue

        return sorted(result)

    def from_train(i, _base):
        """Check if image comes from training dataset"""
        for archive in _base.glob('*.zip'):
            if "TrainingData" not in str(archive):
                continue
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue

                    file = Path(zipinfo.filename)

                    if file.suffix == ".gz" and "seg" not in file.name:
                        if i in str(file):
                            return True
                    else:
                        continue
        return False

    def from_val(i, _base):
        """Check if image comes from validation dataset"""
        for archive in _base.glob('*.zip'):
            if "ValidationData" not in str(archive):
                continue
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue

                    file = Path(zipinfo.filename)

                    if file.suffix == ".gz" and "seg" not in file.name:
                        if i in str(file):
                            return True
                    else:
                        continue
        return False

    @meta
    def mapping21_17(_base) -> pd.DataFrame:
        return pd.read_csv(_base / "BraTS21-17_Mapping.csv")

    def _file(i, _base) -> Tuple[Path, Path]:
        for archive in _base.glob('*.zip'):
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if i == Path(zipinfo.filename).stem.replace(".nii", ""):
                        p = Path(str(zipfile.Path(archive, zipinfo.filename)))
                        archive_path = p.parent.parent.parent
                        relative_path = p.relative_to(archive_path)
                        return archive_path, relative_path

        raise ValueError(f'Id "{i}" not found')

    def subject_id(_file) -> str:
        return _file[1].stem.replace(".nii", "").rsplit("_", 1)[0]

    def modality(_file) -> str:
        return _file[1].stem.replace(".nii", "").rsplit("_", 1)[1]

    def image(_file) -> Union[np.ndarray, None]:
        with unpack(str(_file[0]), str(_file[1]), ".", ".zip") as (unpacked, is_unpacked):
            with open_nii_gz_file(unpacked) as nii_image:
                return np.asarray(nii_image.dataobj)

    def mask(_file) -> Union[np.ndarray, None]:
        mask_postfix = ".".join(["seg", "nii", _file[1].name.split(".")[-1]])
        relative_path = str(_file[1].parent / "_".join([*_file[1].name.split("_")[:-1], mask_postfix]))

        if "Val" in str(_file[0]):
            return None

        with unpack(str(_file[0]), relative_path, ".", ".zip") as (unpacked, is_unpacked):
            with open_nii_gz_file(unpacked) as nii_image:
                return np.asarray(nii_image.dataobj)

    def spacing(_file):
        """Returns pixel spacing along axes (x, y, z)"""
        with unpack(str(_file[0]), str(_file[1]), ".", ".zip") as (unpacked, is_unpacked):
            with open_nii_gz_file(unpacked) as nii_image:
                return tuple(nii_image.header['pixdim'][1:4])

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation"""
        with unpack(str(_file[0]), str(_file[1]), ".", ".zip") as (unpacked, is_unpacked):
            with open_nii_gz_file(unpacked) as nii_image:
                return nii_image.affine
