import nibabel as nb
import numpy as np

from .internals import Dataset, field, register
from .utils import PathOrStr


@register(
    body_region='thorax',
    license=None,  # todo
    link='https://kits-challenge.org/kits23/',
    modality='CT',
    prep_data_size='50G',
    raw_data_size='12G',
    task='Kidney Tumor Segmentation',
)
class KiTS23(Dataset):
    """Kidney and Kidney Tumor Segmentation Challenge,
    The 2023 Kidney and Kidney Tumor Segmentation challenge (abbreviated KiTS23)
    is a competition in which teams compete to develop the best system for
    automatic semantic segmentation of kidneys, renal tumors, and renal cysts.

    Competition page is https://kits-challenge.org/kits23/, official competition repository is
    https://github.com/neheller/kits23/.

    For usage, clone the repository https://github.com/neheller/kits23/, install and run `kits23_download_data`.

    Parameters
    ----------
    root: str, Path
        Absolute path to the root containing the downloaded archive and meta.
        If not provided, the cache is assumed to be already populated.
    """

    def __init__(self, root: PathOrStr):
        super().__init__(root)
        if not (self.root / "dataset").exists():
            raise FileNotFoundError(f"Dataset not found in {self.root}")

    @property
    def ids(self):
        return tuple(sorted(sub.name for sub in (self.root / 'dataset').glob('*')))

    @field
    def image(self, i):
        # CT images are integer-valued, this will help us improve compression rates
        image_file = nb.load(self.root / 'dataset' / i / 'imaging.nii.gz')
        return np.int16(image_file.get_fdata()[...])

    # TODO add multiple segmentations
    @field
    def mask(self, i):
        """Combined annotation for kidneys, tumor and cyst (if present)."""
        ct_scan_nifti = nb.load(self.root / 'dataset' / i / 'segmentation.nii.gz')
        return np.int8(ct_scan_nifti.get_fdata())

    @field
    def affine(self, i):
        """The 4x4 matrix that gives the image's spatial orientation."""
        image_file = nb.load(self.root / 'dataset' / i / 'imaging.nii.gz')
        return image_file.affine

    @property
    def labels_names(self):
        """Indicates which label correspond to which mask, consistent accross all samples."""
        return KITS_LABEL_NAMES


KITS_LABEL_NAMES = {
    # https://github.com/neheller/kits23/blob/063d4c00afd383fc68145a00c0aa6a4e2a3c0f50/kits23/configuration/labels.py#L23
    1: 'kidney',
    2: 'tumor',
    3: 'cyst',
}
