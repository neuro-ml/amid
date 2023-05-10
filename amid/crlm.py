from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import highdicom
import numpy as np
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent
from dicom_csv import get_orientation_matrix, get_slice_locations, get_voxel_spacing, stack_images
from imops import restore_crop
from more_itertools import locate

from .internals import checksum, licenses, register
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
@checksum('crlm')
class CRLM(Source):
    """
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

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

    _root: str = None

    def _base(_root: Silent) -> Path:
        if _root is None:
            raise ValueError('Please pass the locations of the zip archives')
        return Path(_root)

    @meta
    def ids(_base):
        return sorted(d.name for d in _base.iterdir())

    def _folders(i, _base) -> Tuple[Path, Path]:
        case = _base / i
        folders = tuple({p.parent for p in case.glob('*/*/*/*.dcm')})
        return tuple(sorted(folders, key=lambda f: len(list(f.iterdir()))))

    def _series(_folders):
        return series_from_dicom_folder(_folders[1])

    def image(_series):
        return stack_images(_series)

    def mask(image: Output, _series, _folders) -> Dict[str, np.ndarray]:
        """Returns dict: {'liver': ..., 'hepatic': ..., 'tumor_x': ...}"""
        dicom_seg = highdicom.seg.segread(next(_folders[0].glob('*.dcm')))
        image_sops = [s.SOPInstanceUID for s in _series]
        seg_sops = [sop_uid for _, _, sop_uid in dicom_seg.get_source_image_uids()]

        sops = [sop for sop in image_sops if sop in set(seg_sops).intersection(image_sops)]
        seg_box_start = list(locate(image_sops, lambda i: i == sops[0]))[0]
        seg_box_stop = list(locate(image_sops, lambda i: i == sops[-1]))[0]

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

    def spacing(_series):
        """Returns the voxel spacing along axes (x, y, z)."""
        return get_voxel_spacing(_series)

    def slice_locations(_series):
        return get_slice_locations(_series)

    def affine(_series):
        """Returns 4x4 matrix that gives the image's spatial orientation."""
        return get_orientation_matrix(_series)
