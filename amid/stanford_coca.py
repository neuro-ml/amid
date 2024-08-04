import plistlib
import warnings
from ast import literal_eval
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import pydicom
from connectome import Transform
from dicom_csv import (
    drop_duplicated_instances,
    drop_duplicated_slices,
    expand_volumetric,
    get_orientation_matrix,
    get_pixel_spacing,
    get_slice_locations,
    order_series,
    stack_images,
)
from skimage.draw import polygon

from .internals import Dataset, field, licenses, register


class CoCaClasses(IntEnum):
    LAD = 1
    LCX = 2
    RCA = 3
    LCA = 4


class Calcification(NamedTuple):
    label: str
    contour_px: np.ndarray
    contour_mm: np.ndarray
    center: np.ndarray
    area: float
    hu_min: float
    hu_max: float
    hu_dev: float
    hu_mean: float
    length: float
    total: int


@register(
    body_region=('Coronary', 'Chest'),
    license=licenses.StanfordDSResearch,
    link='https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa',
    modality='CT',
    prep_data_size=None,  # TODO: should be measured...
    raw_data_size='28G',
    task='Coronary Calcium Segmentation',
)
class StanfordCoCa(Dataset):
    """
    A Stanford AIMI's Co(ronary) Ca(lcium) dataset.


    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.


    Notes
    -----
    Follow the download instructions at
    https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa.
    You'll need to register and accept the terms of use. After that, copy the files from Azure:

    azcopy copy 'some-generated-access-link' /path/to/downloaded/data/ --recursive=true

    Then, the folder with raw downloaded data should contain two subfolders - a subset with gated coronary CT scans
    and corresponding coronary calcium segmentation masks (`Gated_release_final`)
    and a folder with the non-gated CT scans with corresponding coronary with coronary artery calcium scores
    (`deidentified_nongated`).

    The folder with gated data should have original structure:
        ./Gated_release_final/patient/0/folder-with-dcms/
        ./Gated_release_final/calcium_xml/0.xml
        ...

    The folder with nongated data should have original structure:
        ./deidentified_nongated/0/folder-with-dcms/
        ...

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = StanfordCoCa(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 971
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 57)

    """

    def _split(self, i):
        return i.split('-')[0]

    def _identifier(self, i):
        return i.split('-')[1]

    def _folder_with_images(self, i):
        split = self._split(i)
        if split == 'gated':
            return Path('Gated_release_final') / 'patient'
        if split == 'nongated':
            return 'deidentified_nongated'
        raise ValueError("Unknown split. Use 'gated' or 'nongated' options.")

    def _folder_with_annotations(self, i):
        split = self._split(i)
        if split == 'gated':
            return Path('Gated_release_final') / 'calcium_xml'
        if split == 'nongated':
            return None
        raise ValueError("Unknown split. Use 'gated' or 'nongated' options.")

    @property
    def ids(self):
        gated_ids = tuple(
            sorted('gated-' + x.name for x in (self.root / 'Gated_release_final' / 'patient').iterdir() if x.is_dir())
        )
        nongated_ids = tuple(
            sorted('nongated-' + x.name for x in (self.root / 'deidentified_nongated').iterdir() if x.is_dir())
        )

        return gated_ids + nongated_ids

    def _series(self, i):
        folder_with_dicoms = self.root / self._folder_with_images(i) / self._identifier(i)
        series = list(map(pydicom.dcmread, folder_with_dicoms.glob('*/*.dcm')))
        if not series:
            raise FileNotFoundError(f'No dicoms found at {folder_with_dicoms}')

        # series = sorted(series, key=lambda x: x.InstanceNumber)
        series = expand_volumetric(series)
        series = drop_duplicated_instances(series)

        original_num_slices = len(series)
        series = drop_duplicated_slices(series)
        if len(series) < original_num_slices:
            warnings.warn(f'Dropped duplicated slices for series {series[0]["StudyInstanceUID"]}.')

        series = order_series(series, decreasing=False)

        return series

    @field
    def image(self, i) -> np.ndarray:
        image = stack_images(self._series(i), -1).transpose((1, 0, 2)).astype(np.int16)
        return image

    def _image_meta(self, i):
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
    def series_uid(self, i) -> str:
        return self._image_meta(i).get('SeriesInstanceUID', None)

    @field
    def study_uid(self, i) -> str:
        return self._image_meta(i).get('StudyInstanceUID', None)

    @field
    def pixel_spacing(self, i) -> Union[list, None]:
        series = self._series(i)
        return get_pixel_spacing(series).tolist() if series else None

    @field
    def slice_locations(self, i) -> Union[list, None]:
        series = self._series(i)
        return get_slice_locations(series) if series else None

    @field
    def orientation_matrix(self, i) -> Union[np.ndarray, None]:
        series = self._series(i)
        return get_orientation_matrix(series) if series else None

    def _raw_annotations(self, i):
        """Annotation as it is in xml"""
        folder = self._folder_with_annotations(i)
        if folder is None:
            warnings.warn("The used split doesn't contain segmentation masks.")
            return None

        try:
            with open(self.root / folder / f'{self._identifier(i)}.xml', 'rb') as fp:
                annotation = plistlib.load(fp)
                image_annotations = annotation['Images']

        except FileNotFoundError:
            warnings.warn(f'Missing annotation for id: {i}')
            return None

        return image_annotations

    @field
    def calcifications(self, i) -> Union[list, None]:
        """Returns list of Calcifications"""
        raw_annotations = self._raw_annotations(i)
        if raw_annotations is None:
            return None

        cacs = []
        for slice_annotation in raw_annotations:
            for roi in slice_annotation['ROIs']:
                if roi['Area'] > 0:
                    contour_px = np.array([literal_eval(x) for x in roi['Point_px']])
                    contour_mm = np.array([literal_eval(x) for x in roi['Point_mm']])
                    name, center, area = roi['Name'], roi['Center'], roi['Area']
                    hu_min, hu_max, hu_dev, hu_mean = roi['Min'], roi['Max'], roi['Dev'], roi['Mean']
                    total, length = roi['Total'], roi['Length']
                    cacs.append(
                        Calcification(
                            name, contour_px, contour_mm, center, area, hu_min, hu_max, hu_dev, hu_mean, total, length
                        )
                    )
        return cacs

    @lru_cache(None)
    def _scores(self, i):
        p = self.root / self._folder_with_images(i) / 'scores.xlsx'

        if not p.exists():
            return None

        return pd.read_excel(p, index_col=0)

    @field
    def score(self, i) -> Union[dict, None]:
        scores = self._scores(i)
        if scores is None:
            return None
        try:
            return scores.loc[i + 'A'].to_dict()
        except KeyError:
            warnings.warn(f'Missing scores for idx "{i}"')
            return None


class ContoursToMask(Transform):
    """Our implementation of transform for the contours. One can implement own logic based on this class"""

    __inherit__ = True
    _throw: bool = False
    _class_abbr: dict = {
        'Left Anterior Descending Artery': 'LAD',
        'Left Circumflex Artery': 'LCA',
        'Right Coronary Artery': 'RCA',
        'Left Coronary Artery': 'LCA',
    }

    def mask(id, calcifications, image, slice_locations, _class_abbr, _throw):
        if calcifications is None:
            return None

        shape = image.shape
        sl = slice_locations
        multiclass_mask = np.zeros(shape, np.uint8)
        try:
            for cac in calcifications:
                assert cac.label in _class_abbr, f'Unexpected class: {cac.label}'
                class_name = _class_abbr[cac.label]
                class_id = CoCaClasses[class_name].value
                slice_location = cac.contour_mm[0][-1]
                slice_id = np.argwhere(sl == slice_location).squeeze()
                assert slice_id, 'Slice where calcification is located is not presented in the series.'
                roi_contour = cac.contour_px
                slice_mask = np.zeros(shape[:2])
                xs, ys = polygon(*roi_contour.T, shape[:2])
                slice_mask[xs, ys] = True
                multiclass_mask[..., slice_id] = (class_id * slice_mask.astype(int)).astype(np.uint8)

        except AssertionError as e:
            if _throw:
                raise e
            else:
                warnings.warn(f"Mask preparation for idx {id} failed with: '{str(e)}'. Returning None")
                return None

        return multiclass_mask
