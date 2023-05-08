import os.path
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import mdai
import numpy as np
import pandas as pd
import pydicom
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent
from dicom_csv import (
    drop_duplicated_instances,
    drop_duplicated_slices,
    expand_volumetric,
    get_pixel_spacing,
    get_slice_locations,
    join_tree,
    order_series,
    stack_images,
)
from skimage.draw import polygon

from .internals import checksum, licenses, register


@register(
    body_region='Thorax',
    license=licenses.CC_BYNC_40,
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742',
    modality='CT',
    prep_data_size='7,9G',
    raw_data_size='12G',
    task='COVID-19 Segmentation',
)
@checksum('midrc')
class MIDRC(Source):
    """

        MIDRC-RICORD dataset 1a is a public COVID-19 CT segmentation dataset with 120 scans.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Follow the download instructions at https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742
    Download both Images and Annotations to the same folder

    Then, the folder with downloaded data should contain two paths with the data

    The folder should have this structure:
        <...>/<MIDRC-root>/MIDRC-RICORD-1A
        <...>/<MIDRC-root>/MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json


    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = MIDRC(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
     155
    >>> print(ds.image(ds.ids[0]).shape)
     (512, 512, 112)
    >>> print(ds.mask(ds.ids[80]).shape)
     (6, 512, 512, 450)

    References
    ----------
    """

    _root: str = None
    _pathologies: Tuple[str] = (
        'Infectious opacity',
        'Infectious TIB/micronodules',
        'Atelectasis',
        'Other noninfectious opacity',
        'Noninfectious nodule/mass',
        'Infectious cavity',
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
        joined = join_tree(_base / 'MIDRC-RICORD-1A', verbose=1)
        joined = joined[[x.endswith('.dcm') for x in joined.FileName]]
        joined.to_csv(_base / 'joined.csv')
        return joined

    @meta
    def ids(_joined):
        return tuple(_joined['SeriesInstanceUID'].unique())

    def _annotation(_base):
        json_path = 'MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
        # TODO: do we really need a whole lib to parse one json??? it also generates annoying pandas warning
        return mdai.common_utils.json_to_dataframe(_base / json_path)['annotations']

    def _series(i, _base, _joined):
        sub = _joined[_joined.SeriesInstanceUID == i]
        series_files = sub['PathToFolder'] + os.path.sep + sub['FileName']
        series_files = [_base / 'MIDRC-RICORD-1A' / x for x in series_files]
        series = list(map(pydicom.dcmread, series_files))
        # series = sorted(series, key=lambda x: x.InstanceNumber)
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
        return tuple(spacing.tolist())

    def labels(_study_id, _annotation):
        sub = _annotation[(_annotation.scope == 'STUDY') & (_annotation.StudyInstanceUID == _study_id)]
        return tuple(sub['labelName'].unique())

    def mask(i, image_meta: Output, _annotation, _pathologies):
        # TODO: mask has 6 channels now. Consider adding different methods ot at least a docstring naming channels...
        sub = _annotation[(_annotation.SeriesInstanceUID == i) & (_annotation.scope == 'INSTANCE')]
        shape = (image_meta['Rows'], image_meta['Columns'], len(image_meta['SOPInstanceUID']))
        mask = np.zeros((len(_pathologies), *shape), dtype=bool)
        if len(sub) == 0:
            return None
        for label, row in sub.iterrows():
            pathology_index = _pathologies.index(row['labelName'])
            slice_index = image_meta['SOPInstanceUID'].index(row['SOPInstanceUID'])
            if row['data'] is None:
                warnings.warn(f'{label} annotations for series {i} contains None for slice {slice_index}.')
                continue
            ys, xs = np.array(row['data']['vertices']).T
            mask[(pathology_index, *polygon(ys, xs, shape[:2]), slice_index)] = True
        return mask
