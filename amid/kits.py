from pathlib import Path

import nibabel as nb
import numpy as np
from connectome import Source, Transform, meta
from connectome.interface.nodes import Silent

from .internals import normalize


class KiTS23Base(Source):
    """Kidney and Kidney Tumor Segmentation Challenge,
    The 2023 Kidney and Kidney Tumor Segmentation challenge (abbreviated KiTS23)
    is a competition in which teams compete to develop the best system for
    automatic semantic segmentation of kidneys, renal tumors, and renal cysts.

    Competition page is https://kits-challenge.org/kits23/, official competition repository is
    https://github.com/neheller/kits23/.

    For usage, clone the repository https://github.com/neheller/kits23/, install and run `kits23_download_data`.

    Parameters
    ----------
    root: str, path to downloaded author's repository

    Example
    -------

    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        return tuple(sorted([sub.name for sub in (Path(_root) / 'dataset').glob('*')]))

    def _image_file(i, _root: Silent):
        return nb.load(Path(_root) / 'dataset' / i / 'imaging.nii.gz')

    def image(_image_file):
        # CT images are integer-valued, this will help us improve compression rates
        return np.int16(_image_file.get_fdata()[...])

    # TODO add multiple segmentations
    # TODO add labels mapping
    def mask(i, _root: Silent):
        """ """
        mask_path = Path(_root) / 'dataset' / i / 'segmentation.nii.gz'
        ct_scan_nifti = nb.load(mask_path)
        return np.bool_(ct_scan_nifti.get_fdata()[...])

    def affine(_image_file):
        """The 4x4 matrix that gives the image's spatial orientation."""
        return _image_file.affine


class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nb.affines.voxel_sizes(affine)


KiTS23 = normalize(
    KiTS23Base,
    'KiTS23',
    'kits',
    body_region='thorax',
    license=None,  # todo
    link='https://kits-challenge.org/kits23/',
    modality='CT',
    prep_data_size='50G',
    raw_data_size='12G',
    task='Kidney Tumor Segmentation',
    normalizers=[SpacingFromAffine()],
)
