from pathlib import Path

import nibabel as nb
import numpy as np
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent
from deli import load

from .internals import checksum, register
from .utils import deprecate


@register(
    body_region='Head',
    license='EGD data license',
    link='https://xnat.bmia.nl/data/archive/projects/egd',
    modality=('FLAIR', 'MRI T1', 'MRI T1GD', 'MRI T2'),
    prep_data_size='107,49G',
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

    def _base(_root: Silent):
        if _root is None:
            raise ValueError('Please provide the `root` argument')
        return Path(_root)

    @meta
    def ids(_base):
        result = []
        for folder in (_base / 'SUBJECTS').iterdir():
            for suffix in 'FLAIR', 'T1', 'T1GD', 'T2':
                result.append(f'{folder.name}-{suffix}')

        return tuple(sorted(result))

    def brain_mask(i, _base):
        return np.bool_(nb.load(_base / 'METADATA' / 'Brain_mask.nii.gz').get_fdata())

    def deface_mask(i, _base):
        return np.bool_(nb.load(_base / 'METADATA' / 'Deface_mask.nii.gz').get_fdata())

    def _image_file(i, _base):
        i, suffix = i.rsplit('-', 1)
        return nb.load(_base / 'SUBJECTS' / i / f'{suffix}.nii.gz')

    def modality(i):
        _, suffix = i.rsplit('-', 1)
        return suffix

    def subject_id(i):
        subject, _ = i.rsplit('-', 1)
        return subject

    def affine(_image_file):
        return _image_file.affine

    @deprecate(message='Use `spacing` method instead.')
    def voxel_spacing(spacing: Output):
        return spacing

    def spacing(_image_file):
        # voxel spacing is [1, 1, 1] for all images in this dataset...
        return tuple(_image_file.header['pixdim'][1:4])

    def image(_image_file):
        # intensities are not integer-valued in this dataset...
        return np.asarray(_image_file.dataobj)

    def _metadata(i, _base):
        i, _ = i.rsplit('-', 1)
        return load(_base / 'SUBJECTS' / i / 'metadata.json')

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

    def mask(i, _base):
        i, _ = i.rsplit('-', 1)
        return np.bool_(nb.load(_base / 'SUBJECTS' / i / 'MASK.nii.gz').get_fdata())
