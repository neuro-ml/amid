import json
import os
from pathlib import Path

import nibabel as nb
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from .internals import checksum, register


@register(
    body_region='Head',
    license='EGD data license',
    link='https://www.sciencedirect.com/science/article/pii/S2352340921004753',
    modality=('FLAIR', 'MRI T1', 'MRI T1GD', 'MRI T2'),
    prep_data_size=None,  # TODO: should be measured...
    raw_data_size='40G',
    task='Segmentation',
)
@checksum('egd')
class EGD(Source):
    """
    The Erasmus Glioma Database (EGD): Structural MRI scans, WHO 2016 subtypes,
    and segmentations of 774 patients with glioma [1]_.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    The access to the dataset could be requested at XNAT portal [https://xnat.bmia.nl/data/archive/projects/egd].

    To download the data in the compatible structure we recommend to use
    egd-downloader script [https://zenodo.org/record/4761089#.YtZpLtJBxhF].
    Please, refer to its README for further information.

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> egd = EGD(root='/path/to/downloaded/data/folder/')
    >>> print(len(egd.ids))
    # 774
    >>> print(egd.t1gd(egd.ids[215]).shape)
    # (197, 233, 189)
    >>> print(egd.manufacturer(egd.ids[444]))
    # Philips Medical Systems

    References
    ----------
    .. [1] van der Voort, Sebastian R., et al. "The Erasmus Glioma Database (EGD): Structural MRI scans,
           WHO 2016 subtypes, and segmentations of 774 patients with glioma."
           Data in brief 37 (2021): 107191.
           https://www.sciencedirect.com/science/article/pii/S2352340921004753

    """

    _root: str = None

    @meta
    def ids(_root):
        return tuple(sorted(set(os.listdir(Path(_root) / 'SUBJECTS'))))

    @meta
    def brain_mask(_root):
        return np.bool_(nb.load(Path(_root) / 'METADATA' / 'Brain_mask.nii.gz').get_fdata())

    @meta
    def deface_mask(_root):
        return np.bool_(nb.load(Path(_root) / 'METADATA' / 'Deface_mask.nii.gz').get_fdata())

    def _image_file(i, _root: Silent):
        return nb.load(Path(_root) / 'SUBJECTS' / i / 'FLAIR.nii.gz')

    def affine(_image_file):
        return _image_file.affine

    def voxel_spacing(_image_file):
        # voxel spacing is [1, 1, 1] for all images in this dataset...
        return tuple(_image_file.header['pixdim'][1:4])

    def flair(_image_file):
        # intensities are not integer-valued in this dataset...
        return np.asarray(_image_file.dataobj)

    def t1(i, _root: Silent):
        # intensities are not integer-valued in this dataset...
        return np.asarray(nb.load(Path(_root) / 'SUBJECTS' / i / 'T1.nii.gz').dataobj)

    def t1gd(i, _root: Silent):
        # intensities are not integer-valued in this dataset...
        return np.asarray(nb.load(Path(_root) / 'SUBJECTS' / i / 'T1GD.nii.gz').dataobj)

    def t2(i, _root: Silent):
        # intensities are not integer-valued in this dataset...
        return np.asarray(nb.load(Path(_root) / 'SUBJECTS' / i / 'T2.nii.gz').dataobj)

    def _metadata(i, _root: Silent):
        with open(Path(_root) / 'SUBJECTS' / i / 'metadata.json', 'r') as f:
            return json.load(f)

    def genetic_and_histological_label_idh(_metadata):
        return _metadata['Genetic_and_Histological_labels']['IDH']

    def genetic_and_histological_label_1p19q(_metadata):
        return _metadata['Genetic_and_Histological_labels']['1p19q']

    def genetic_and_histological_label_grade(_metadata):
        return _metadata['Genetic_and_Histological_labels']['Grade']

    def age(_metadata):
        return _metadata['Clinical_data']['Age']

    def sex(_metadata):
        return _metadata['Clinical_data']['Sex']

    def observer(_metadata):
        return _metadata['Segmentation_source']['Observer']

    def original_scan(_metadata):
        return _metadata['Segmentation_source']['Original scan']

    def manufacturer(_metadata):
        return _metadata['Scan_characteristics']['Manufacturer']

    def system(_metadata):
        return _metadata['Scan_characteristics']['System']

    def field(_metadata):
        return _metadata['Scan_characteristics']['Field']

    def mask(i, _root: Silent):
        return np.bool_(nb.load(Path(_root) / 'SUBJECTS' / i / 'MASK.nii.gz').get_fdata())
