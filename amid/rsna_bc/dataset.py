from contextlib import suppress
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pydicom
from bev.utils import PathOrStr
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import checksum, register
from .utils import add_csv_fields, fields, unpack


@register(
    body_region='Thorax',
    license='Non-Commercial Use',
    link='https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data',
    modality='MG',
    raw_data_size='271G',
    prep_data_size='294G',
    task='Breast cancer classification',
)
@checksum('rsna-breast-cancer', cache_columns=tuple(fields))
class RSNABreastCancer(Source):
    _root: PathOrStr

    def _base(_root: Silent):
        return Path(_root)

    @lru_cache(None)
    def _meta(_base):
        dfs = []
        for part in 'train', 'test':
            with suppress(FileNotFoundError):
                with unpack(_base, f'{part}.csv') as (file, _):
                    df = pd.read_csv(file)
                    df['part'] = part
                    dfs.append(df)

        if not dfs:
            raise FileNotFoundError('No metadata found')
        dfs = pd.concat(dfs, ignore_index=True)
        for field in 'image_id', 'patient_id', 'site_id':
            dfs[field] = dfs[field].astype(str)

        raw = list(map(str, dfs.image_id.tolist()))
        ids = set(raw)
        if len(ids) != len(raw):
            raise ValueError('The image ids are not unique')

        return {row.image_id: row for _, row in dfs.iterrows()}

    def _row(i, _meta):
        return _meta[i]

    add_csv_fields(locals())

    @meta
    def ids(_meta):
        return tuple(sorted(_meta))

    def _dicom(_row, _base):
        with unpack(_base, f'{_row.part}_images/{_row.patient_id}/{_row.image_id}.dcm') as (file, _):
            return pydicom.dcmread(file)

    def image(_dicom):
        return _dicom.pixel_array

    def padding_value(_dicom):
        return getattr(_dicom, 'PixelPaddingValue', None)

    def intensity_sign(_dicom):
        return getattr(_dicom, 'PixelIntensityRelationshipSign', None)
