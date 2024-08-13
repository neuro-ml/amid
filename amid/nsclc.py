import json
import warnings
from functools import cached_property
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pydicom
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

from .internals import Dataset, field, licenses, register


@register(
    body_region='Thorax',
    license=licenses.CC_BY_30,
    link='https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics',
    modality='CT',
    prep_data_size='13G',
    raw_data_size='34G',
    task='Tumor Segmentation',
)
class NSCLC(Dataset):
    """

        NSCLC-Radiomics is a public cell lung cancer segmentation dataset with 422 patients.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.


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

    @cached_property
    def _joined(self):
        joined_path = self.root / 'joined.csv'
        if joined_path.exists():
            return pd.read_csv(joined_path)
        joined = join_tree(self.root / 'NSCLC-Radiomics', verbose=1)
        joined = joined[[x.endswith('.dcm') for x in joined.FileName]]
        joined.to_csv(joined_path)
        return joined

    @property
    def ids(self):
        uid = self._joined.groupby('SeriesInstanceUID').apply(len)
        return tuple(uid[uid > 1].keys())

    def _sub(self, i):
        return self._joined[self._joined.SeriesInstanceUID == i]

    def _series(self, i):
        series = [
            pydicom.dcmread(self.root / 'NSCLC-Radiomics' / file.PathToFolder / file.FileName)
            for _, file in self._sub(i).iterrows()
        ]
        series = expand_volumetric(series)
        series = drop_duplicated_instances(series)

        if True:  # drop_dupl_slices
            _original_num_slices = len(series)
            series = drop_duplicated_slices(series)
            if len(series) < _original_num_slices:
                warnings.warn(f'Dropped duplicated slices for series {series[0]["StudyInstanceUID"]}.')

        return order_series(series, decreasing=False)

    @field
    def image(self, i) -> np.ndarray:
        image = stack_images(self._series(i), -1).astype(np.int16).transpose(1, 0, 2)
        return image

    @field
    def image_meta(self, i) -> dict:
        metas = [list(dict(s).values()) for s in self._series(i)]
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

    @field
    def sex(self, i) -> str:
        """Sex of the patient."""
        return self._sub(i)['PatientSex'].iloc[1]

    @field
    def age(self, i) -> Union[int, None]:
        """Age of the patient, dataset contains 97 patients with unknown Age."""
        age = self._sub(i)['PatientAge'].iloc[1]
        if isinstance(age, str):
            return int(age.removesuffix('Y'))
        return age

    def _study_id(self, i):
        study_ids = self._joined[self._joined.SeriesInstanceUID == i].StudyInstanceUID.unique()
        assert len(study_ids) == 1
        # series_id_to_study
        return study_ids[0]

    @field
    def spacing(self, i) -> np.ndarray:
        pixel_spacing = get_pixel_spacing(self._series(i)).tolist()
        slice_locations = get_slice_locations(self._series(i))
        diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
        spacing = np.float32([pixel_spacing[1], pixel_spacing[0], diffs[np.argsort(counts)[-1]]])
        return spacing

    @field
    def mask(self, i) -> np.ndarray:
        return self._extract_segment_masks(i).get('GTV-1', None)

    @field
    def lung_left(self, i) -> np.ndarray:
        return self._extract_segment_masks(i).get('Lung-Left', None)

    @field
    def lung_right(self, i) -> np.ndarray:
        return self._extract_segment_masks(i).get('Lung-Right', None)

    @field
    def lungs_total(self, i) -> np.ndarray:
        return self._extract_segment_masks(i).get('Lungs-Total', None)

    @field
    def heart(self, i) -> np.ndarray:
        return self._extract_segment_masks(i).get('Heart', None)

    @field
    def esophagus(self, i) -> np.ndarray:
        return self._extract_segment_masks(i).get('Esophagus', None)

    @field
    def spinal_cord(self, i) -> np.ndarray:
        return self._extract_segment_masks(i).get('Spinal-Cord', None)

    def _extract_segment_masks(self, i):
        folders = self._joined[self._joined.SeriesInstanceUID == i].PathToFolder.unique()
        assert len(folders) == 1, i
        patient_id = self._joined[self._joined.SeriesInstanceUID == i].PatientID.unique()[0]
        if patient_id in self._INVALID_PATIENT_IDS:
            return {}

        annotation_path = self.root / 'NSCLC-Radiomics' / Path(folders[0]).parent

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
        assert np.allclose(get_orientation_matrix(self._series(i)), get_cancer_orientation_matrix(cancer_dicom)), i
        mask = np.moveaxis(cancer_dicom.pixel_array, 0, -1).astype(bool).transpose(1, 0, 2)
        mask_slice_locations = get_mask_slice_locations(cancer_dicom)
        slice_locations = get_slice_locations(self._series(i))
        image = stack_images(self._series(i), -1).transpose(1, 0, 2)
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
