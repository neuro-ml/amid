import contextlib
import zipfile
from gzip import GzipFile
from pathlib import Path

import nibabel


@contextlib.contextmanager
def unpack(root: str, relative: str, archive_root_name: str = None, archive_ext: str = None):
    """Provides the absolute path to the file in both scenarios: inside archive or inside folder.

    Parameters
    ----------
    root : str, Path
        Absolute path to the downloaded archive or the unpacked archive root.
    relative : str, Path
        Relative file path inside the archive. Archive's root folder sholud be ommited.
    archive_root_name : str, Path, optional
        If `root` is a archive, it's root folder name shold be given.
    archive_ext: {'.zip'}, optional
        Compression algorithm used to create the archive

    Returns
    -------
    unpacked : Path
        Absolute file path to be opened.
    is_unpacked : {True, False}
        Reached file state. `True` if the file is located inside archive, `False` otherwise.
    """
    unpacked = Path(root) / relative

    if unpacked.exists():
        yield unpacked, True
    elif archive_ext == '.zip':
        with zipfile.Path(root, str(Path(archive_root_name, relative))).open('rb') as unpacked:
            yield unpacked, False
    else:
        raise ValueError('Unexpected file path or unsupported compression algorithm.')


@contextlib.contextmanager
def open_nii_gz_file(unpacked):
    """Opens ``.nii.gz`` file if it is packed in archive

    Examples
    --------
    >>> with unpack('/path/to/archive.zip', 'relative/file/path', 'root', '.zip') as (unpacked, is_unpacked):
    >>>     with open_nii_gz_file(unpacked) as image:
    >>>         print(np.asarray(image.dataobj).shape)
    # (512, 512, 256)
    """
    with GzipFile(fileobj=unpacked) as nii:
        nii = nibabel.FileHolder(fileobj=nii)
        yield nibabel.Nifti1Image.from_file_map({'header': nii, 'image': nii})
