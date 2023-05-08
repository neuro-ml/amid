import json
import os.path
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from connectome import Source, meta
from connectome.interface.nodes import Silent
from dicom_csv import (
    drop_duplicated_instances,
    drop_duplicated_slices,
    expand_volumetric,
    get_orientation_matrix,
    get_pixel_spacing,
    get_slice_locations,
    join_tree,
    order_series,
    stack_images,
)

from .internals import checksum, licenses, register


@register(
    body_region='Thorax',
    license=licenses.CC_BY_30,
    link='https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics',
    modality='CT',
    prep_data_size='13G',
    raw_data_size='34G',
    task='Tumor Segmentation',
)
@checksum('nsclc')
class NSCLC(Source):
    """

        NSCLC-Radiomics is a public cell lung cancer segmentation dataset with 422 patients.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Follow the download instructions at https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics

    The folder with downloaded data should contain two paths

    The folder should have this structure:
        <...>/<NSCLC-root>/NSCLC-Radiomics/LUNG1-XXX


    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = NSCLC(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
     422
    >>> print(ds.image(ds.ids[0]).shape)
     (512, 512, 134)
    >>> print(ds.mask(ds.ids[80]).shape)
     (512, 512, 108)

    References
    ----------
    """

    _root: str = None

    # FIXME: move to filtering via `ignore_errors=True` + filtering via Filter (if needed)
    _INVALID_PATIENT_IDS = (
        # no dicom with cancer segmentation
        'LUNG1-128',
        'LUNG1-412',
        # image.shape != cancer.shape
        'LUNG1-194',
        'LUNG1-095',
        'LUNG1-085',
        'LUNG1-014',
        'LUNG1-021',
    )

    def _base(_root: Silent):
        if _root is None:
            raise ValueError('Please provide the `root` argument')
        return Path(_root)

    @lru_cache(None)
    def _joined(_base):
        joined_path = _base / 'joined.csv'
        if joined_path.exists():
            return pd.read_csv(joined_path)
        joined = join_tree(_base / 'NSCLC-Radiomics', verbose=1)
        joined = joined[[x.endswith('.dcm') for x in joined.FileName]]
        joined.to_csv(joined_path)
        return joined

    @meta
    def ids(_joined):
        uid = _joined.groupby('SeriesInstanceUID').apply(len)
        return tuple(uid[uid > 1].keys())

    def _series(i, _base, _joined):
        sub = _joined[_joined.SeriesInstanceUID == i]
        series_files = sub['PathToFolder'] + os.path.sep + sub['FileName']
        series_files = [_base / 'NSCLC-Radiomics' / x for x in series_files]
        series = list(map(pydicom.dcmread, series_files))
        series = expand_volumetric(series)
        series = drop_duplicated_instances(series)

        if True:  # drop_dupl_slices
            _original_num_slices = len(series)
            series = drop_duplicated_slices(series)
            if len(series) < _original_num_slices:
                warnings.warn(f'Dropped duplicated slices for series {series[0]["StudyInstanceUID"]}.')

        series = order_series(series, decreasing=False)
        return series

    def image(_series):
        image = stack_images(_series, -1).astype(np.int16).transpose(1, 0, 2)
        return image

    def image_meta(_series):
        metas = [list(dict(s).values()) for s in _series]
        result = {}
        for meta_ in metas:
            for element in meta_:
                if element.keyword in ['PixelData']:
                    continue
                if element.keyword not in result:
                    result[element.keyword] = [element.value]
                elif result[element.keyword][-1] != element.value:
                    result[element.keyword].append(element.value)
        # turn elements that are the same across the series back from array
        result = {k: v[0] if len(v) == 1 else v for k, v in result.items()}
        return result

    def _study_id(i, _joined):
        study_ids = _joined[_joined.SeriesInstanceUID == i].StudyInstanceUID.unique()
        assert len(study_ids) == 1
        # series_id_to_study
        return study_ids[0]

    def spacing(_series):
        pixel_spacing = get_pixel_spacing(_series).tolist()
        slice_locations = get_slice_locations(_series)
        diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
        spacing = np.float32([pixel_spacing[1], pixel_spacing[0], diffs[np.argsort(counts)[-1]]])
        return spacing

    def mask(_extract_segment_masks):
        return _extract_segment_masks.get('GTV-1', None)

    def lung_left(_extract_segment_masks):
        return _extract_segment_masks.get('Lung-Left', None)

    def lung_right(_extract_segment_masks):
        return _extract_segment_masks.get('Lung-Right', None)

    def lungs_total(_extract_segment_masks):
        return _extract_segment_masks.get('Lungs-Total', None)

    def heart(_extract_segment_masks):
        return _extract_segment_masks.get('Heart', None)

    def esophagus(_extract_segment_masks):
        return _extract_segment_masks.get('Esophagus', None)

    def spinal_cord(_extract_segment_masks):
        return _extract_segment_masks.get('Spinal-Cord', None)

    def _extract_segment_masks(i, _series, _joined, _base, _INVALID_PATIENT_IDS):
        folders = _joined[_joined.SeriesInstanceUID == i].PathToFolder.unique()
        assert len(folders) == 1, i
        patient_id = _joined[_joined.SeriesInstanceUID == i].PatientID.unique()[0]
        if patient_id in _INVALID_PATIENT_IDS:
            return {}

        annotation_path = _base / 'NSCLC-Radiomics' / Path(folders[0]).parent

        found_markup = None
        for p in annotation_path.glob('*.json'):
            with open(p, 'r') as f:
                data = json.load(f)
            if 'Segmentation' in data['Total'][-1]:
                assert found_markup is None, annotation_path
                found_markup = data

        if found_markup is None:
            raise FileNotFoundError

        dicom_pathes = list((annotation_path / found_markup['SeriesUID']).glob('*'))
        assert len(dicom_pathes) == 1, annotation_path
        cancer_dicom = pydicom.dcmread(dicom_pathes[0])
        assert np.allclose(get_orientation_matrix(_series), get_cancer_orientation_matrix(cancer_dicom)), i
        mask = np.moveaxis(cancer_dicom.pixel_array, 0, -1).astype(bool).transpose(1, 0, 2)
        mask_slice_locations = get_mask_slice_locations(cancer_dicom)
        slice_locations = get_slice_locations(_series)
        image = stack_images(_series, -1).transpose(1, 0, 2)
        segments = [x.SegmentDescription for x in cancer_dicom.SegmentSequence]
        assert len(mask_slice_locations) == len(slice_locations) * len(segments), i

        all_masks = {}
        for n, seg in enumerate(segments):
            mask_subslice = slice(
                len(slice_locations) * n, len(slice_locations) * (n + 1) if n + 1 != len(segments) else None
            )
            sub_mask = mask[:, :, mask_subslice]
            sub_mask_slice_locations = mask_slice_locations[mask_subslice]

            assert sub_mask.shape == image.shape, i
            if np.allclose(sub_mask_slice_locations, slice_locations, atol=0.01):
                pass
            elif np.allclose(sub_mask_slice_locations, slice_locations[::-1], atol=0.01):
                sub_mask = sub_mask[..., ::-1]
            else:
                raise AssertionError(i)
            all_masks[seg] = sub_mask
        return all_masks


def get_cancer_orientation_matrix(cancer_dicom):
    row, col = np.array(
        [
            float(x)
            for x in cancer_dicom.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
        ]
    ).reshape(2, 3)
    return np.stack([row, col, np.cross(row, col)])


def get_mask_slice_locations(cancer_dicom):
    om = get_cancer_orientation_matrix(cancer_dicom)
    image_position_patient = np.stack(
        [
            list(map(float, frame.PlanePositionSequence[0].ImagePositionPatient))
            for frame in cancer_dicom.PerFrameFunctionalGroupsSequence
        ]
    )
    return list(image_position_patient @ om[-1])
