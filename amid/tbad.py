import gzip
from pathlib import Path

import nibabel as nb
import numpy as np
from connectome import Source, Transform, meta
from connectome.interface.nodes import Silent

from .internals import licenses, normalize


class TBADBase(Source):
    """
    A dataset of 3D Computed Tomography (CT) images for Type-B Aortic Dissection segmentation.

    Notes
    -----
    The data can only be obtained by contacting the authors by email.
    See the [dataset home page](https://github.com/XiaoweiXu/Dataset_Type-B-Aortic-Dissection) for details.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded files.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Examples
    --------
    >>> # Place the downloaded files in any folder and pass the path to the constructor:
    >>> ds = TBAD(root='/path/to/files/root')
    >>> print(len(ds.ids))
    # 100
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 327)

    References
    ----------
    .. [1] Yao, Zeyang & Xie, Wen & Zhang, Jiawei & Dong, Yuhao & Qiu, Hailong & Haiyun, Yuan & Jia,
           Qianjun & Tianchen, Wang & Shi, Yiyi & Zhuang, Jian & Que, Lifeng & Xu, Xiaowei & Huang, Meiping.
           (2021). ImageTBAD: A 3D Computed Tomography Angiography Image Dataset for Automatic Segmentation
           of Type-B Aortic Dissection. Frontiers in Physiology. 12. 732711. 10.3389/fphys.2021.732711.
    """

    _root: str = None

    def _base_p(_root: Silent) -> Path:
        if _root is None:
            raise ValueError('Please pass the path to the root folder')
        return Path(_root)

    @meta
    def ids(_base_p: Silent):
        result = set()

        for file in Path(_base_p).glob('*_image.nii.gz'):
            result.add(file.stem.split('_')[0])

        return tuple(sorted(result))

    def _fname(i, _base_p: Silent):
        return Path(_base_p / f'{i}_image.nii.gz')

    def image(_fname):
        with _fname.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.int16(image.get_fdata())

    def affine(_fname):
        """The 4x4 matrix that gives the image's spatial orientation."""
        with _fname.open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    def mask(i, _base_p: Silent):
        with Path(_base_p / f'{i}_label.nii.gz').open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                label = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.uint8(label.get_fdata())


class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nb.affines.voxel_sizes(affine)


TBAD = normalize(
    TBADBase,
    'TBAD',
    'tbad',
    body_region='Chest',
    license=licenses.CC_BYNC_40,
    link='https://github.com/XiaoweiXu/Dataset_Type-B-Aortic-Dissection',
    modality='CT',
    prep_data_size='14G',
    raw_data_size='14G',
    task='Aortic dissection segmentation',
    normalizers=[SpacingFromAffine()],
)
