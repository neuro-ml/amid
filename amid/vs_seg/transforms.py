from typing import Sequence, Union

import numpy as np
from connectome import Transform
from imops import zoom

from ..utils import propagate_none


Numeric = Union[float, int]


class CanonicalMRIOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.transpose(image, (1, 0, 2))[..., ::-1]

    def spacing(spacing):
        return tuple(np.array(spacing)[[1, 0, 2]].tolist())

    @propagate_none
    def schwannoma(schwannoma):
        return np.transpose(schwannoma, (1, 0, 2))[..., ::-1]

    @propagate_none
    def cochlea(cochlea):
        return np.transpose(cochlea, (1, 0, 2))[..., ::-1]

    @propagate_none
    def meningioma(meningioma):
        return np.transpose(meningioma, (1, 0, 2))[..., ::-1]


class Rescale(Transform):
    __inherit__ = True

    _new_spacing: Union[Sequence[Numeric], Numeric]
    _order: int = 1

    def _spacing(spacing, _new_spacing):
        _new_spacing = np.broadcast_to(_new_spacing, len(spacing)).copy()
        _new_spacing[np.isnan(_new_spacing)] = np.array(spacing)[np.isnan(_new_spacing)]
        return tuple(_new_spacing.tolist())

    def _scale_factor(spacing, _spacing):
        return np.float32(spacing) / np.float32(_spacing)

    def image(image, _scale_factor, _order):
        return zoom(image.astype(np.float32), _scale_factor, order=_order)

    @propagate_none
    def schwannoma(schwannoma, _scale_factor, _order):
        return zoom(schwannoma.astype(np.float32), _scale_factor, order=_order) > 0.5

    @propagate_none
    def cochlea(cochlea, _scale_factor, _order):
        return zoom(cochlea.astype(np.float32), _scale_factor, order=_order) > 0.5

    @propagate_none
    def meningioma(meningioma, _scale_factor, _order):
        return zoom(meningioma.astype(np.float32), _scale_factor, order=_order) > 0.5
