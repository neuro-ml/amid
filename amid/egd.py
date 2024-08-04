import nibabel as nb
import numpy as np
from deli import load

from .internals import Dataset, field as _field, register


@register(
    body_region='Head',
    license='EGD data license',
    link='https://xnat.bmia.nl/data/archive/projects/egd',
    modality=('FLAIR', 'MRI T1', 'MRI T1GD', 'MRI T2'),
    prep_data_size='107,49G',
    raw_data_size='40G',
    task='Segmentation',
)
class EGD(Dataset):
    """
    The Erasmus Glioma Database (EGD): Structural MRI scans, WHO 2016 subtypes,
    and segmentations of 774 patients with glioma [1]_.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

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

    @property
    def ids(self):
        result = []
        for folder in (self.root / 'SUBJECTS').iterdir():
            for suffix in 'FLAIR', 'T1', 'T1GD', 'T2':
                result.append(f'{folder.name}-{suffix}')

        return tuple(sorted(result))

    @_field
    def brain_mask(self, i) -> np.ndarray:
        return nb.load(self.root / 'METADATA' / 'Brain_mask.nii.gz').get_fdata().astype(bool)

    @_field
    def deface_mask(self, i) -> np.ndarray:
        return nb.load(self.root / 'METADATA' / 'Deface_mask.nii.gz').get_fdata().astype(bool)

    def _image_file(self, i):
        i, suffix = i.rsplit('-', 1)
        return nb.load(self.root / 'SUBJECTS' / i / f'{suffix}.nii.gz')

    @_field
    def modality(self, i) -> str:
        _, suffix = i.rsplit('-', 1)
        return suffix

    @_field
    def subject_id(self, i) -> str:
        subject, _ = i.rsplit('-', 1)
        return subject

    @_field
    def affine(self, i) -> np.ndarray:
        return self._image_file(i).affine

    def spacing(self, i):
        # voxel spacing is [1, 1, 1] for all images in this dataset...
        return tuple(self._image_file(i).header['pixdim'][1:4])

    @_field
    def image(self, i) -> np.ndarray:
        # intensities are not integer-valued in this dataset...
        return np.asarray(self._image_file(i).dataobj)

    def _metadata(self, i):
        i, _ = i.rsplit('-', 1)
        return load(self.root / 'SUBJECTS' / i / 'metadata.json')

    @_field
    def genetic_and_histological_label_idh(self, i) -> str:
        return self._metadata(i)['Genetic_and_Histological_labels']['IDH']

    @_field
    def genetic_and_histological_label_1p19q(self, i) -> str:
        return self._metadata(i)['Genetic_and_Histological_labels']['1p19q']

    @_field
    def genetic_and_histological_label_grade(self, i) -> str:
        return self._metadata(i)['Genetic_and_Histological_labels']['Grade']

    @_field
    def age(self, i) -> float:
        return self._metadata(i)['Clinical_data']['Age']

    @_field
    def sex(self, i) -> str:
        return self._metadata(i)['Clinical_data']['Sex']

    @_field
    def observer(self, i) -> str:
        return self._metadata(i)['Segmentation_source']['Observer']

    @_field
    def original_scan(self, i) -> str:
        return self._metadata(i)['Segmentation_source']['Original scan']

    @_field
    def manufacturer(self, i) -> str:
        return self._metadata(i)['Scan_characteristics']['Manufacturer']

    @_field
    def system(self, i) -> str:
        return self._metadata(i)['Scan_characteristics']['System']

    @_field
    def field(self, i) -> str:
        return self._metadata(i)['Scan_characteristics']['Field']

    @_field
    def mask(self, i) -> np.ndarray:
        i, _ = i.rsplit('-', 1)
        return nb.load(self.root / 'SUBJECTS' / i / 'MASK.nii.gz').get_fdata().astype(bool)
