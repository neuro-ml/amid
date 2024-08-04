import json
import os.path
import warnings
from functools import cached_property
from typing import Tuple

import numpy as np
import pandas as pd
import pydicom
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

from .internals import Dataset, field, licenses, register


@register(
    body_region='Thorax',
    license=licenses.CC_BYNC_40,
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742',
    modality='CT',
    prep_data_size='7,9G',
    raw_data_size='12G',
    task='COVID-19 Segmentation',
)
class MIDRC(Dataset):
    """

        MIDRC-RICORD dataset 1a is a public COVID-19 CT segmentation dataset with 120 scans.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.

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

    _fields = 'image', 'image_meta', 'spacing', 'labels', 'mask'

    @cached_property
    def _joined(self):
        joined_path = self.root / 'joined.csv'
        if joined_path.exists():
            joined = pd.read_csv(joined_path)
        else:
            joined = join_tree(self.root / 'MIDRC-RICORD-1A', verbose=1)
            joined = joined[[x.endswith('.dcm') for x in joined.FileName]]
            joined.to_csv(self.root / 'joined.csv', index=False)
        return joined

    @cached_property
    def ids(self):
        return tuple(self._joined['SeriesInstanceUID'].unique())

    @cached_property
    def _annotation(self):
        json_path = 'MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
        return json_to_dataframe(self.root / json_path)['annotations']

    def _series(self, i):
        sub = self._joined[self._joined.SeriesInstanceUID == i]
        series_files = sub['PathToFolder'] + os.path.sep + sub['FileName']
        series_files = [self.root / 'MIDRC-RICORD-1A' / x for x in series_files]
        series = list(map(pydicom.dcmread, series_files))
        # series = sorted(series, key=lambda x: x.InstanceNumber)
        series = expand_volumetric(series)
        series = drop_duplicated_instances(series)

        # if drop_dupl_slices:
        _original_num_slices = len(series)
        series = drop_duplicated_slices(series)
        if len(series) < _original_num_slices:
            warnings.warn(f'Dropped duplicated slices for series {series[0]["StudyInstanceUID"]}.')

        series = order_series(series, decreasing=False)
        return series

    @field
    def image(self, i):
        image = stack_images(self._series(i), -1).astype(np.int16).transpose(1, 0, 2)
        return image

    @field
    def image_meta(self, i):
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

    def _study_id(self, i):
        study_ids = self._joined[self._joined.SeriesInstanceUID == i].StudyInstanceUID.unique()
        assert len(study_ids) == 1
        # series_id_to_study
        return study_ids[0]

    @field
    def spacing(self, i):
        series = self._series(i)
        pixel_spacing = get_pixel_spacing(series).tolist()
        slice_locations = get_slice_locations(series)
        diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
        spacing = np.float32([pixel_spacing[1], pixel_spacing[0], diffs[np.argsort(counts)[-1]]])
        return tuple(spacing.tolist())

    @field
    def labels(self, i):
        sub = self._annotation[
            (self._annotation.scope == 'STUDY') & (self._annotation.StudyInstanceUID == self._study_id(i))
        ]
        return tuple(sub['labelName'].unique())

    @field
    def mask(self, i):
        # TODO: mask has 6 channels now. Consider adding different methods ot at least a docstring naming channels...
        sub = self._annotation[(self._annotation.SeriesInstanceUID == i) & (self._annotation.scope == 'INSTANCE')]
        image_meta = self.image_meta(i)

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


_pathologies: Tuple[str, ...] = (
    'Infectious opacity',
    'Infectious TIB/micronodules',
    'Atelectasis',
    'Other noninfectious opacity',
    'Noninfectious nodule/mass',
    'Infectious cavity',
)


# TODO: simplify
def json_to_dataframe(json_file, datasets=None):
    if datasets is None:
        datasets = []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    a = pd.DataFrame([])
    studies = pd.DataFrame([])
    labels = None

    # Gets annotations for all datasets
    for d in data['datasets']:
        if d['id'] in datasets or len(datasets) == 0:
            study = pd.DataFrame(d['studies'])
            study['dataset'] = d['name']
            study['datasetId'] = d['id']
            studies = pd.concat([studies, study], ignore_index=True, sort=False)

            annots = pd.DataFrame(d['annotations'])
            annots['dataset'] = d['name']
            a = pd.concat([a, annots], ignore_index=True, sort=False)

    if len(studies) > 0:
        studies = studies[['StudyInstanceUID', 'dataset', 'datasetId', 'number']]
    g = pd.DataFrame(data['labelGroups'])
    # unpack arrays
    result = pd.DataFrame([(d, tup.id, tup.name) for tup in g.itertuples() for d in tup.labels])
    if len(result) > 0:
        result.columns = ['labels', 'labelGroupId', 'labelGroupName']

        def unpack_dictionary(df, column):
            ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].items()))], axis=1, sort=False)
            del ret[column]
            return ret

        labels = unpack_dictionary(result, 'labels')
        if 'parentId' in labels.columns:
            labels = labels[
                [
                    'labelGroupId',
                    'labelGroupName',
                    'annotationMode',
                    'color',
                    'description',
                    'id',
                    'name',
                    'radlexTagIds',
                    'scope',
                    'parentId',
                ]
            ]
            labels.columns = [
                'labelGroupId',
                'labelGroupName',
                'annotationMode',
                'color',
                'description',
                'labelId',
                'labelName',
                'radlexTagIdsLabel',
                'scope',
                'parentLabelId',
            ]
        else:
            labels = labels[
                [
                    'labelGroupId',
                    'labelGroupName',
                    'annotationMode',
                    'color',
                    'description',
                    'id',
                    'name',
                    'radlexTagIds',
                    'scope',
                ]
            ]
            labels.columns = [
                'labelGroupId',
                'labelGroupName',
                'annotationMode',
                'color',
                'description',
                'labelId',
                'labelName',
                'radlexTagIdsLabel',
                'scope',
            ]

        if len(a) > 0:
            a = a.merge(labels, on=['labelId'], sort=False)
    if len(studies) > 0 and len(a) > 0:
        a = a.merge(studies, on=['StudyInstanceUID', 'dataset'], sort=False)
        # Format data
        studies.number = studies.number.astype(int)
        a.number = a.number.astype(int)
        a.loc.createdAt = pd.to_datetime(a.createdAt)
        a.loc.updatedAt = pd.to_datetime(a.updatedAt)
    return {'annotations': a, 'studies': studies, 'labels': labels}
