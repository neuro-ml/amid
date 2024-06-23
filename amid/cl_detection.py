from functools import cached_property
from typing import Dict, Tuple

import numpy as np
import SimpleITK
from connectome import Transform
from deli import load
from imops import crop_to_box

from .internals import Dataset, field, licenses, register
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
class CLDetection2023(Dataset):
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

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = CLDetection2023(root='/path/to/data/root/folder')
    >>> print(len(ds.ids))
    # 400
    >>> print(ds.image(ds.ids[0]).shape)
    # (2400, 1935)
    """

    @cached_property
    def _images(self):
        return SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(self.root / 'train_stack.mha'))

    @cached_property
    def _points(self):
        return load(self.root / 'train-gt.json')['points']

    @property
    def ids(self):
        return tuple(map(str, range(1, len(self._images) + 1)))

    @field
    def image(self, i) -> np.ndarray:
        i = int(i)
        return self._images[i - 1]

    @field
    def points(self, i) -> Dict[str, np.ndarray]:
        i = int(i)
        return {x['name']: np.array(x['point'][:2]) for x in self._points if x['point'][-1] == i}

    @field
    def spacing(self, i) -> Tuple[float, float]:
        i = int(i)
        (scale,) = {x['scale'] for x in self._points if x['point'][-1] == i}
        scale = float(scale)
        return scale, scale


class CropPadding(Transform):
    __inherit__ = 'spacing'

    def _box(image):
        return mask_to_box(image[..., 0] != 0)

    def image(image, _box):
        return crop_to_box(image[..., 0], _box)

    def points(points, _box):
        return {k: v - _box[0] for k, v in points.items()}


class FlipPoints(Transform):
    __inherit__ = True

    def points(points):
        return {name: pt[::-1] for name, pt in points.items()}
