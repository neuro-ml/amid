from typing import Sequence, Union

import numpy as np
from connectome import Transform
from imops import zoom

from ..utils import Numeric


class CanonicalCTOrientation(Transform):
    __exclude__ = ('nodules', 'nodules_masks')

    def image(image):
        return image[..., ::-1]

    def cancer(cancer):
        return cancer[..., ::-1]


class Rescale(Transform):
    __exclude__ = ('pixel_spacing', 'slice_locations', 'voxel_spacing', 'orientation_matrix')

    _new_spacing: Union[Sequence[Numeric], Numeric]
    _order: int = 1

    def _spacing(spacing, _new_spacing):
        _new_spacing = np.broadcast_to(_new_spacing, len(spacing)).copy()
        _new_spacing[np.isnan(_new_spacing)] = np.array(spacing)[np.isnan(_new_spacing)]
        return tuple(_new_spacing.tolist())

    def _scale_factor(spacing, _spacing):
        return np.float32(spacing) / np.float32(_spacing)

    def spacing(_spacing):
        return _spacing

    def image(image, _scale_factor, _order):
        return zoom(image.astype(np.float32), _scale_factor, order=_order)

    def cancer(cancer, _scale_factor, _order):
        return zoom(cancer.astype(np.float32), _scale_factor, order=_order) > 0.5
