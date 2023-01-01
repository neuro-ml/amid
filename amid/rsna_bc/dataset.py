from functools import lru_cache
from pathlib import Path

import pandas as pd
import pydicom
from bev.utils import PathOrStr
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import checksum, register
from .utils import add_csv_fields


@register(
    body_region='Thorax',
    license='Non-Commercial Use',
    link='https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data',
    modality='MG',
    raw_data_size='271G',
    prep_data_size='294G',
    task='Breast cancer classification',
)
@checksum('rsna-breast-cancer')
class RSNABreastCancer(Source):
    _root: PathOrStr

    def _base(_root: Silent):
        return Path(_root)

    @lru_cache(None)
    def _meta(_base):
        dfs = []
        for part in 'train', 'test':
            if (_base / f'{part}.csv').exists():
                df = pd.read_csv(_base / f'{part}.csv')
                df['part'] = part
                dfs.append(df)

        if not dfs:
            raise FileNotFoundError('No metadata found')
        dfs = pd.concat(dfs, ignore_index=True)
        for field in 'image_id', 'patient_id', 'site_id':
            dfs[field] = dfs[field].astype(str)
        return dfs

    def _row(i, _meta):
        row = _meta[_meta.image_id == i]
        assert len(row) == 1
        return row.iloc[0]

    add_csv_fields(locals())

    @meta
    def ids(_meta):
        raw = list(map(str, _meta.image_id.tolist()))
        ids = tuple(sorted(set(raw)))
        if len(ids) != len(raw):
            raise ValueError('The image ids are not unique')
        return ids

    def _dicom(_row, _base):
        return pydicom.dcmread(_base / f'{_row.part}_images' / _row.patient_id / f'{_row.image_id}.dcm')

    def image(_dicom):
        return _dicom.pixel_array

    def padding_value(_dicom):
        return getattr(_dicom, 'PixelPaddingValue', None)

    def intensity_sign(_dicom):
        return getattr(_dicom, 'PixelIntensityRelationshipSign', None)
