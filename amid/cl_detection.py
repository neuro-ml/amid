from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK
from connectome import Source, Transform, meta
from connectome.interface.nodes import Silent
from deli import load
from imops import crop_to_box

from .internals import checksum, licenses, register
from .utils import mask_to_box


@register(
    body_region='Head',
    license=licenses.CC_BYNC_40,
    link='https://github.com/cwwang1979/CL-detection2023/',
    modality='X-ray',
    prep_data_size='1.8G',
    raw_data_size='1.5G',
    task='Keypoint detection',
)
@checksum('cl-detection-2023')
class CLDetection2023(Source):
    """
    The data for the "Cephalometric Landmark Detection in Lateral X-ray Images" Challenge,
    held with the MICCAI-2023 conference.

    Notes
    -----
    The data can only be obtained by contacting the organizers by email.
    See the [challenge home page](https://cl-detection2023.grand-challenge.org/) for details.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded and unarchived data.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = CLDetection2023(root='/path/to/data/root/folder')
    >>> print(len(ds.ids))
    # 400
    >>> print(ds.image(ds.ids[0]).shape)
    # (2400, 1935)
    """

    _root: str = None

    def _base(_root: Silent) -> Path:
        if _root is None:
            raise ValueError('Please pass the path to the root folder')
        return Path(_root)

    @lru_cache(1)
    def _images(_base):
        return SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(_base / 'train_stack.mha'))

    @lru_cache(1)
    def _points(_base):
        return load(_base / 'train-gt.json')['points']

    @meta
    def ids(_images):
        return tuple(map(str, range(1, len(_images) + 1)))

    def image(i, _images):
        i = int(i)
        return _images[i - 1]

    def points(i, _points):
        i = int(i)
        return {x['name']: np.array(x['point'][:2]) for x in _points if x['point'][-1] == i}

    def spacing(i, _points):
        i = int(i)
        (scale,) = {x['scale'] for x in _points if x['point'][-1] == i}
        scale = float(scale)
        return [scale, scale]

    @classmethod
    def normalizer(cls):
        return CropPadding()


class CropPadding(Transform):
    __inherit__ = 'spacing'

    def _box(image):
        return mask_to_box(image[..., 0] != 0)

    def image(image, _box):
        return crop_to_box(image[..., 0], _box)

    def points(points, _box):
        return {k: v - _box[0] for k, v in points.items()}
