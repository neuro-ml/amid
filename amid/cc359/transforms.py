from typing import Sequence, Union

import numpy as np
from connectome import Transform
from imops import zoom

from ..utils import Numeric, propagate_none


class CanonicalMRIOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.transpose(image, (1, 0, 2))[::-1, :, ::-1]

    def spacing(spacing):
        return tuple(np.array(spacing)[[1, 0, 2]].tolist())

    def brain(brain):
        return np.transpose(brain, (1, 0, 2))[::-1, :, ::-1]

    @propagate_none
    def hippocampus(hippocampus):
        return np.transpose(hippocampus, (1, 0, 2))[::-1, :, ::-1]

    @propagate_none
    def wm_gm_csf(wm_gm_csf):
        return np.transpose(wm_gm_csf, (1, 0, 2))[::-1, :, ::-1]


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

    def spacing(_spacing):
        return _spacing

    def image(image, _scale_factor, _order):
        return zoom(image.astype(np.float32), _scale_factor, order=_order)

    def brain(brain, _scale_factor, _order):
        return zoom(brain.astype(np.float32), _scale_factor, order=_order) > 0.5

    @propagate_none
    def hippocampus(hippocampus, _scale_factor, _order):
        return zoom(hippocampus.astype(np.float32), _scale_factor, order=_order) > 0.5

    @propagate_none
    def wm_gm_csf(wm_gm_csf, _scale_factor, _order):
        onehot = np.arange(wm_gm_csf.max() + 1) == wm_gm_csf[..., None]
        onehot = onehot.astype(wm_gm_csf.dtype).transpose(3, 0, 1, 2)
        out = np.array(zoom(onehot.astype(np.float32), _scale_factor, axis=(1, 2, 3)) > 0.5, dtype=wm_gm_csf.dtype)
        labels = out.argmax(axis=0)
        return labels
