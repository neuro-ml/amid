from contextlib import suppress
from functools import cached_property

import pandas as pd
import pydicom

from ..internals import Dataset, field, register
from .utils import csv_field, unpack


@register(
    body_region='Thorax',
    license='Non-Commercial Use',
    link='https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data',
    modality='MG',
    raw_data_size='271G',
    prep_data_size='294G',
    task='Breast cancer classification',
)
class RSNABreastCancer(Dataset):
    @cached_property
    def _meta(self):
        dfs = []
        for part in 'train', 'test':
            with suppress(FileNotFoundError):
                with unpack(self.root, f'{part}.csv') as (file, _):
                    df = pd.read_csv(file)
                    df['part'] = part
                    dfs.append(df)

        if not dfs:
            raise FileNotFoundError('No metadata found')
        dfs = pd.concat(dfs, ignore_index=True)
        for name in 'image_id', 'patient_id', 'site_id':
            dfs[name] = dfs[name].astype(str)

        raw = list(map(str, dfs.image_id.tolist()))
        ids = set(raw)
        if len(ids) != len(raw):
            raise ValueError('The image ids are not unique')

        return {row.image_id: row for _, row in dfs.iterrows()}

    # csv fields
    site_id = csv_field('site_id', str)
    patient_id = csv_field('patient_id', str)
    image_id = csv_field('image_id', str)
    laterality = csv_field('laterality', None)
    view = csv_field('view', None)
    age = csv_field('age', None)
    cancer = csv_field('cancer', bool)
    biopsy = csv_field('biopsy', bool)
    invasive = csv_field('invasive', bool)
    BIRADS = csv_field('BIRADS', int)
    implant = csv_field('implant', bool)
    density = csv_field('density', None)
    machine_id = csv_field('machine_id', str)
    prediction_id = csv_field('prediction_id', str)
    difficult_negative_case = csv_field('difficult_negative_case', bool)

    @property
    def ids(self):
        return tuple(sorted(self._meta))

    def _dicom(self, i):
        row = self._meta[i]
        with unpack(self.root, f'{row.part}_images/{row.patient_id}/{row.image_id}.dcm') as (file, _):
            return pydicom.dcmread(file)

    @field
    def image(self, i):
        return self._dicom(i).pixel_array

    @field
    def padding_value(self, i):
        return getattr(self._dicom(i), 'PixelPaddingValue', None)

    @field
    def intensity_sign(self, i):
        return getattr(self._dicom(i), 'PixelIntensityRelationshipSign', None)
