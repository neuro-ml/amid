import numpy as np
from pylidc import Annotation

from .typing import (
    Calcification,
    InternalStructure,
    LIDCNodule,
    Lobulation,
    Malignancy,
    Sphericity,
    Spiculation,
    Subtlety,
    Texture,
)


def get_nodule(ann: Annotation) -> LIDCNodule:
    def init_enum(enum_class, value):
        try:
            return enum_class(value)
        except ValueError:
            pass

    bbox = ann.bbox_matrix().T
    bbox[1] = bbox[1] + 1

    return LIDCNodule(
        center_voxel=ann.centroid,
        bbox=bbox,
        diameter_mm=ann.diameter,
        surface_area_mm2=ann.surface_area,
        volume_mm3=ann.volume,
        calcification=init_enum(Calcification, ann.calcification),
        internal_structure=init_enum(InternalStructure, ann.internalStructure),
        lobulation=init_enum(Lobulation, ann.lobulation),
        malignancy=init_enum(Malignancy, ann.malignancy),
        sphericity=init_enum(Sphericity, ann.sphericity),
        spiculation=init_enum(Spiculation, ann.spiculation),
        subtlety=init_enum(Subtlety, ann.subtlety),
        texture=init_enum(Texture, ann.texture),
    )


def flip_nodule(nodule: LIDCNodule, n_slices: int) -> LIDCNodule:
    bbox = nodule.bbox.copy()
    start_slice, stop_slice = bbox[:, -1]
    bbox[:, -1] = np.array([n_slices - stop_slice, n_slices - start_slice])

    center_voxel = nodule.center_voxel
    center_voxel[-1] = n_slices - center_voxel[-1]

    return nodule._replace(
        center_voxel=center_voxel,
        bbox=bbox,
    )
