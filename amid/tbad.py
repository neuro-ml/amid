import gzip
from pathlib import Path

import nibabel as nb
import numpy as np

from .internals import Dataset, field, licenses, register


@register(
    body_region='Chest',
    license=licenses.CC_BYNC_40,
    link='https://github.com/XiaoweiXu/Dataset_Type-B-Aortic-Dissection',
    modality='CT',
    prep_data_size='14G',
    raw_data_size='14G',
    task='Aortic dissection segmentation',
)
class TBAD(Dataset):
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

    @property
    def ids(self):
        result = set()

        for file in self.root.glob('*_image.nii.gz'):
            result.add(file.stem.split('_')[0])

        return tuple(sorted(result))

    def _fname(self, i):
        return self.root / f'{i}_image.nii.gz'

    def image(self, i) -> np.ndarray:
        with self._fname(i).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.int16(image.get_fdata())

    def affine(self, i) -> np.ndarray:
        """The 4x4 matrix that gives the image's spatial orientation."""
        with self._fname(i).open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    @field
    def mask(self, i) -> np.ndarray:
        with Path(self.root / f'{i}_label.nii.gz').open('rb') as opened:
            with gzip.GzipFile(fileobj=opened) as nii:
                nii = nb.FileHolder(fileobj=nii)
                label = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.uint8(label.get_fdata())
