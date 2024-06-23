from functools import partial
from typing import Dict

import highdicom
import numpy as np
from dicom_csv import get_orientation_matrix, get_slice_locations, get_voxel_spacing, stack_images
from imops import restore_crop
from more_itertools import locate

from .internals import Dataset, licenses, register
from .utils import series_from_dicom_folder


@register(
    body_region='Abdomen',
    license=licenses.CC_BY_40,
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?'
    'pageId=89096268#89096268412b832037484784bd78caf58e052641',
    modality=('CT, SEG'),
    prep_data_size='11G',
    raw_data_size='11G',
    task=('Segmentation', 'Classification'),
)
class CRLM(Dataset):
    """
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.


    Notes
    -----
    Download links:
    https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096268#89096268b2cc35fce0664a2b875b5ec675ba9446

    This collection consists of DICOM images and DICOM Segmentation Objects (DSOs)
    for 197 patients with Colorectal Liver Metastases (CRLM).
    Comprised of Original DICOM CTs and Segmentations for each subject.
    The segmentations include 'Liver', 'Liver_Remnant'
    (liver that will remain after surgery based on a preoperative CT plan),
    'Hepatic' and 'Portal' veins,
    and 'Tumor_x', where 'x' denotes the various tumor occurrences in the case

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = CRLM(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 197
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 52)

    References
    ----------
    """

    @property
    def ids(self):
        return sorted(d.name for d in self.root.iterdir())

    def _folders(self, i):
        case = self.root / i
        folders = tuple({p.parent for p in case.glob('*/*/*/*.dcm')})
        return tuple(sorted(folders, key=lambda f: len(list(f.iterdir()))))

    def _series(self, i):
        return series_from_dicom_folder(self._folders(i)[1])

    def image(self, i):
        return stack_images(self._series(i))

    def mask(self, i) -> Dict[str, np.ndarray]:
        """Returns dict: {'liver': ..., 'hepatic': ..., 'tumor_x': ...}"""
        dicom_seg = highdicom.seg.segread(next(self._folders(i)[0].glob('*.dcm')))
        series = self._series(i)
        image_sops = [s.SOPInstanceUID for s in series]
        seg_sops = [sop_uid for _, _, sop_uid in dicom_seg.get_source_image_uids()]

        sops = [sop for sop in image_sops if sop in set(seg_sops).intersection(image_sops)]
        seg_box_start = list(locate(image_sops, lambda i: i == sops[0]))[0]
        seg_box_stop = list(locate(image_sops, lambda i: i == sops[-1]))[0]

        image = self.image(i)
        seg_box = np.asarray(((0, 0, seg_box_start), (*np.atleast_1d(image.shape[:-1]), seg_box_stop + 1)))

        raw_masks = np.swapaxes(
            dicom_seg.get_pixels_by_source_instance(
                sops,
                ignore_spatial_locations=True,
                segment_numbers=dicom_seg.get_segment_numbers(),
            ),
            -1,
            0,
        )
        masks = list(map(partial(restore_crop, box=seg_box, shape=image.shape), raw_masks))

        liver_mask = {'liver': masks[0].astype(bool)}
        # skip liver remnant
        veins = {'hepatic': masks[2].astype(bool), 'portal': masks[3].astype(bool)}
        tumors = {f'tumor_{i}': array.astype(bool) for i, array in enumerate(masks[4:])}

        return {**liver_mask, **veins, **tumors}

    def spacing(self, i):
        """Returns the voxel spacing along axes (x, y, z)."""
        return get_voxel_spacing(self._series(i))

    def slice_locations(self, i):
        return get_slice_locations(self._series(i))

    def affine(self, i):
        """Returns 4x4 matrix that gives the image's spatial orientation."""
        return get_orientation_matrix(self._series(i))
