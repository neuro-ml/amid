import numpy as np
from connectome import Transform


class CanonicalMRIOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.transpose(image, (1, 0, 2))[..., ::-1]

    def spacing(spacing):
        return tuple(np.array(spacing)[[1, 0, 2]].tolist())

    def schwannoma(schwannoma):
        return None if (schwannoma is None) else np.transpose(schwannoma, (1, 0, 2))[..., ::-1]

    def cochlea(cochlea):
        return None if (cochlea is None) else np.transpose(cochlea, (1, 0, 2))[..., ::-1]

    def meningioma(meningioma):
        return None if (meningioma is None) else np.transpose(meningioma, (1, 0, 2))[..., ::-1]
