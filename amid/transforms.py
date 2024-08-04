import nibabel
import numpy as np
from connectome import Output, Transform


class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nibabel.affines.voxel_sizes(affine)


class ParseAffineMatrix(Transform):
    """Splits affine matrix into separate methods for more convenient usage.

    Examples
    --------
    >>> dataset = Dataset()
    >>> dataset.voxel_spacing(id_)
    # FieldError
    >>> dataset = dataset >> ParseAffineMatrix()
    >>> dataset.voxel_spacing(id_)
    # array([1.5, 1.5, 1.5])
    """

    __inherit__ = True

    def origin(affine):
        """Constructs an origin tensor from the given affine matrix."""
        return affine[:-1, -1]

    def spacing(affine):
        """Constructs a voxel spacing tensor from the given orientation matrix."""
        return np.linalg.norm(affine[:3, :3], axis=0)

    def orientation(affine, spacing: Output):
        """Constructs an orientation matrix from the given affine matrix."""
        return np.divide(affine[:3, :3], spacing)
