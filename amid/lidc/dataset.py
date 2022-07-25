import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent, Output
import pylidc as pl
from dicom_csv import (expand_volumetric, drop_duplicated_instances,
                       drop_duplicated_slices, order_series, stack_images,
                       get_slice_locations, get_pixel_spacing, get_tag,
                       get_orientation_matrix, get_common_tag)

from amid.internals import checksum, register
from amid.cancer_500.dataset import _get_study_date
from .nodules import get_nodule


@register(
    body_region='Chest',
    modality='CT',
    task='lung nodule segmentation',
    licence='TCIA Data Usage Policy and Creative Commons Attribution 3.0 Unported License'
)
@checksum('lidc')
class LIDC(Source):
    """
    The (L)ung (I)mage (D)atabase (C)onsortium image collection (LIDC-IDRI) [1]_ 
    consists of diagnostic and lung cancer screening thoracic computed tomography (CT) scans 
    with marked-up annotated lesions and lung nodules segmentation task.
    Scans contains multiple expert annotations.

    Number of CT scans: 1018.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Follow the download instructions at https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI.

    Then, the folder with raw downloaded data should contain folder `LIDC-IDRI`, 
    which contains folders `LIDC-IDRI-*`.

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = LIDC(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 1018
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 194)
    >>> print(ds.cancer(ds.ids[0]).shape)
    # (512, 512, 194)

    References
    ----------
    .. [1] Armato III, McLennan, et al. "The lung image database consortium (lidc) and image database 
    resource initiative (idri): a completed reference database of lung nodules on ct scans." 
    Medical physics 38(2) (2011): 915–931.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041807/
    """

    _root: str = None

    @meta
    def ids(_root: Silent):
        result = {f'lidc_{scan.series_instance_uid}' for scan in pl.query(pl.Scan).all()}
        return tuple(sorted(result))

    def _scan(i, _root: Silent):
        _id = i.split('_')[-1]
        return pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == _id).first()

    def _series(_scan):
        series = expand_volumetric(_scan.load_all_dicom_images(verbose=False))
        series = order_series(series)
        return series

    def _shape(_series):
        return stack_images(_series, -1).shape

    def image(_scan):
        return _scan.to_volume(verbose=False)

    def study_uid(_scan):
        return _scan.study_instance_uid

    def series_uid(_scan):
        return _scan.series_instance_uid

    def patient_id(_scan):
        return _scan.patient_id

    def sop_uids(_series):
        return [str(get_tag(i, 'SOPInstanceUID')) for i in _series]

    def pixel_spacing(_scan):
        spacing = _scan.pixel_spacing
        return [spacing, spacing]

    def slice_locations(_scan):
        return _scan.slice_zvals

    def voxel_spacing(_scan, pixel_spacing: Output):
        """ Returns voxel spacing along axes (x, y, z). """
        spacing = np.float32([pixel_spacing[0], pixel_spacing[0], _scan.slice_spacing])
        return spacing

    def contrast_used(_scan):
        """ If the DICOM file for the scan had any Contrast tag, this is marked as `True`. """
        return _scan.contrast_used

    def is_from_initial(_scan):
        """
        Indicates whether or not this PatientID was tagged as 
        part of the initial 399 release.
        """
        return _scan.is_from_initial

    def orientation_matrix(_series):
        return get_orientation_matrix(_series)

    def conv_kernel(_series):
        return get_common_tag(_series, 'ConvolutionKernel', default=None)

    def kvp(_series):
        return get_common_tag(_series, 'KVP', default=None)

    def study_date(_series):
        return _get_study_date(_series)

    def accession_number(_series):
        return get_common_tag(_series, 'AccessionNumber', default=None)

    def nodules(_scan):
        nodules = []
        for anns in _scan.cluster_annotations():
            nodule_annotations = []
            for ann in anns:
                nodule_annotations.append(get_nodule(ann))
            nodules.append(nodule_annotations)
        return nodules

    def nodules_masks(_scan):
        nodules = []
        for anns in _scan.cluster_annotations():
            nodule_annotations = []
            for ann in anns:
                nodule_annotations.append(ann.boolean_mask())
            nodules.append(nodule_annotations)
        return nodules

    def cancer(_scan, _shape):
        cancer = np.zeros(_shape, dtype=bool)
        for nodule_index, anns in enumerate(_scan.cluster_annotations()):
            cancer |= pl.utils.consensus(anns, pad=np.inf)[0]

        return cancer
