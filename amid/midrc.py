import os.path
import warnings
from functools import lru_cache
from skimage.draw import polygon
from pathlib import Path

import mdai
import numpy as np
import pandas as pd
import pydicom
from connectome import Source, meta
from connectome.interface.nodes import Silent
from .internals import checksum
from dicom_csv import (expand_volumetric, drop_duplicated_instances, 
                       drop_duplicated_slices, order_series, stack_images, 
                       get_slice_locations, get_pixel_spacing, get_tag, join_tree)

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

    Then, the folder with downloaded data should contain two pathes with the data

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
    _pathologies: [str] = ['Infectious opacity',
                           'Infectious TIB/micronodules',
                           'Atelectasis',
                           'Other noninfectious opacity',
                           'Noninfectious nodule/mass',
                           'Infectious cavity']

    @meta
    def ids(_joined):
        return tuple(_joined['SeriesInstanceUID'].unique())

    @lru_cache(None)
    def _joined(_root: Silent):
        if os.path.exists(Path(_root) / "joined.csv"):
            return pd.read_csv(Path(_root) / "joined.csv")
        joined = join_tree(Path(_root) / 'MIDRC-RICORD-1A', verbose=1)
        joined = joined[[x.endswith('.dcm') for x in joined.FileName]]
        joined.to_csv(Path(_root) / "joined.csv")
        return joined

    def _annotation(_root: Silent):
        json_path = "MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json"
        return mdai.common_utils.json_to_dataframe(Path(_root) / json_path)['annotations']

    def _series(i, _root: Silent, _joined):
        sub = _joined[_joined.SeriesInstanceUID == i]
        series_files = sub['PathToFolder'] + os.path.sep + sub['FileName']
        series_files = [Path(_root)  / 'MIDRC-RICORD-1A' / x for x in series_files]
        series = list(map(pydicom.dcmread, series_files))
        #series = sorted(series, key=lambda x: x.InstanceNumber)
        series = expand_volumetric(series)
        series = drop_duplicated_instances(series)

        if True: # drop_dupl_slices
            _original_num_slices = len(series)
            series = drop_duplicated_slices(series)
            if len(series) < _original_num_slices:
                warnings.warn(f'Dropped duplicated slices for series {_series[0]["StudyInstanceUID"]}.')

        series = order_series(series)
        return series

    def image(_series):
        image = stack_images(_series, -1).astype(np.int16)
        return image

    def _image_meta(_series):
        metas = [list(dict(s).values()) for s in _series]
        result = {}
        for meta in metas:
            for element in meta:
                if element.keyword in ['PixelData']:
                    continue
                if element.keyword not in result:
                    result[element.keyword] = [element.value]
                elif result[element.keyword][-1] != element.value:
                    result[element.keyword].append(element.value)
        # turn elements that are the same across the series back from array
        result = {k: v[0] if len(v) == 1 else v for k, v in result.items()}
        return result

    def image_meta(_image_meta):
        return _image_meta

    def _study_id(i, _joined):
        study_ids = _joined[_joined.SeriesInstanceUID == i].StudyInstanceUID.unique()
        assert len(study_ids) == 1
        # series_id_to_study
        return study_ids[0]

    def voxel_spacing(_series):
        pixel_spacing = get_pixel_spacing(_series).tolist()
        slice_locations = get_slice_locations(_series)
        diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
        spacing = np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])
        return spacing

    def labels(_study_id, _annotation):
        sub = _annotation[(_annotation.scope == "STUDY")
                          & (_annotation.StudyInstanceUID == _study_id)]
        return tuple(sub['labelName'].unique())

    def mask(i, _image_meta, _annotation, _pathologies):
        sub = _annotation[(_annotation.SeriesInstanceUID == i) & (_annotation.scope == "INSTANCE")]
        shape = (_image_meta['Rows'], _image_meta['Columns'], len(_image_meta['SOPInstanceUID']))
        mask = np.zeros((len(_pathologies), *shape), dtype=bool)

        for label, row in sub.iterrows():
            pathology_index = _pathologies.index(row['labelName'])
            slice_index = _image_meta['SOPInstanceUID'].index(row['SOPInstanceUID'])
            if row['data'] is None:
                warnings.warn(f'{label} annotations for series {i} contains None for slice {slice_index}.')
                continue
            ys, xs = np.array(row['data']['vertices']).T[::-1]
            mask[(pathology_index, *polygon(ys, xs, shape[:2]), slice_index)] = True
        return mask



