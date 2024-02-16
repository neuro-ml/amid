import zipfile
from pathlib import Path
from zipfile import ZipFile

import nibabel as nb
import numpy as np
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent

from ..internals import licenses, normalize
from ..utils import deprecate


class LiTSBase(Source):
    """
    A (Li)ver (T)umor (S)egmentation dataset [1]_ from Medical Segmentation Decathlon [2]_

    There are two segmentation tasks on this dataset: liver and liver tumor segmentation.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Follow the download instructions at https://competitions.codalab.org/competitions/17094.

    Then, the folder with raw downloaded data should contain two zip archives with the train data
    (`Training_Batch1.zip` and `Training_Batch2.zip`)
    and a folder with the test data
    (`LITS-Challenge-Test-Data`).

    The folder with test data should have original structure:
        <...>/LITS-Challenge-Test-Data/test-volume-0.nii
        <...>/LITS-Challenge-Test-Data/test-volume-1.nii
        ...

    P.S. Organs boxes are also provided from a separate source https://github.com/superxuang/caffe_3d_faster_rcnn.

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = LiTS(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 201
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 163)
    >>> print(ds.tumor_mask(ds.ids[80]).shape)
    # (512, 512, 771)

    References
    ----------
    .. [1] Bilic, Patrick, et al. "The liver tumor segmentation benchmark (lits)."
           arXiv preprint arXiv:1901.04056 (2019).
    .. [2] Antonelli, Michela, et al. "The medical segmentation decathlon."
           arXiv preprint arXiv:2106.05735 (2021).
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        result = set()
        # zip archives for train images:
        for archive in Path(_root).glob('*.zip'):
            with ZipFile(archive) as zf:
                for zipinfo in zf.infolist():
                    if zipinfo.is_dir():
                        continue

                    file_stem = Path(zipinfo.filename).stem
                    if 'volume' in file_stem:
                        result.add('lits-train-' + file_stem.split('-')[-1])

        # folder for test images:
        for file in (Path(_root) / 'LITS-Challenge-Test-Data').glob('*'):
            result.add('lits-test-' + file.stem.split('-')[-1])

        return tuple(sorted(result))

    def fold(i, _root: Silent):
        num_id = i.split('-')[-1]

        if 'train' in i:
            for archive in Path(_root).glob('*.zip'):
                batch = '1' if ('1' in archive.stem) else '2'

                with ZipFile(archive) as zf:
                    for zipinfo in zf.infolist():
                        if zipinfo.is_dir():
                            continue

                        if num_id == Path(zipinfo.filename).stem.split('-')[-1]:
                            return f'train_batch_{batch}'

        else:  # if 'test' in i:
            return 'test'

    def _file(i, _root: Silent):
        num_id = i.split('-')[-1]

        if 'train' in i:
            for archive in Path(_root).glob('*.zip'):
                with ZipFile(archive) as zf:
                    for zipinfo in zf.infolist():
                        if zipinfo.is_dir():
                            continue

                        file = Path(zipinfo.filename)
                        if ('volume' in file.stem) and (num_id == file.stem.split('-')[-1]):
                            return zipfile.Path(str(archive), str(file))

        else:  # if 'test' in i:
            return Path(_root) / 'LITS-Challenge-Test-Data' / f'test-volume-{num_id}.nii'

        raise KeyError(f'Id "{i}" not found')

    def image(_file):
        with _file.open('rb') as nii:
            nii = nb.FileHolder(fileobj=nii)
            image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
            # most ct scans are integer-valued, this will help us improve compression rates
            return np.int16(image.get_fdata())

    def affine(_file):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with _file.open('rb') as nii:
            nii = nb.FileHolder(fileobj=nii)
            image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
            return image.affine

    @deprecate(message='Use `spacing` method instead.')
    def voxel_spacing(spacing: Output):
        return spacing

    def spacing(_file):
        """Returns voxel spacing along axes (x, y, z)."""
        with _file.open('rb') as nii:
            nii = nb.FileHolder(fileobj=nii)
            image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
            return tuple(image.header['pixdim'][1:4])

    def mask(_file):
        if 'test' not in _file.name:
            with (_file.parent / _file.name.replace('volume', 'segmentation')).open('rb') as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.uint8(image.get_fdata())


LiTS = normalize(
    LiTSBase,
    'LiTS',
    'lits',
    body_region='Abdominal',
    license=licenses.CC_BYNCND_40,
    link='https://competitions.codalab.org/competitions/17094',
    modality='CT',
    prep_data_size='24,7G',
    raw_data_size='35G',
    task='Segmentation',
)
