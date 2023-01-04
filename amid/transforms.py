import numpy as np
from connectome import Output, Transform


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

    def orientation(affine):
        """Constructs an orientation matrix from the given affine matrix."""
        return affine[:3, :3]

    def origin(affine):
        """Constructs an origin tensor from the given affine matrix."""
        return affine[:-1, -1]

    def voxel_spacing(orientation: Output):
        """Constructs a voxel spacing tensor from the given orientation matrix."""
        return np.linalg.norm(orientation, axis=0)
