from datetime import datetime
from functools import cached_property
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .internals import Dataset, field, licenses, register


class LUNA25Nodule(NamedTuple):
    coords: Sequence[float]
    lesion_id: int
    annotation_id: str
    nodule_id: str
    malignancy: bool
    center_voxel: Sequence[float]
    bbox: np.ndarray


@register(
    body_region='Chest',
    license=licenses.CC_BY_40,
    link='https://luna25.grand-challenge.org/',
    modality='CT',
    prep_data_size='214G',
    raw_data_size='205G',
    task='Lung nodule malignancy risk estimation',
)
class LUNA25(Dataset):
    """
    The LUNA25 Challenge dataset is a comprehensive collection designed to support
    the development and validation of AI algorithms for lung nodule malignancy risk
    estimation using low-dose chest CT scans. In total, it contains 2120 patients
    and 4069 low-dose chest CT scans, with 555 annotated malignant nodules and
    5608 benign nodules (3762 unique nodules, 348 of them are malignant).
    The dataset was acquired in participants who enrolled in the
    National Lung Cancer Screening Trial (NLST) between 2002 and 2004 in
    one of the 33 centers in the United States.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing `luna25_images` and `luna25_nodule_blocks` folders and
        `LUNA25_Public_Training_Development_Data.csv` file obtained by the instruction at
        https://luna25.grand-challenge.org/datasets/.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Join the challenge at https://luna25.grand-challenge.org/.
    Then follow the download and extraction instructions at https://luna25.grand-challenge.org/datasets/.
    """

    @property
    def ids(self):
        return [file.name[: -len('.mha')] for file in (self.root / 'luna25_images').iterdir()]

    def _sitk_image(self, i):
        return sitk.ReadImage(self.root / f'luna25_images/{i}.mha')

    @field
    def image(self, i):
        return sitk.GetArrayFromImage(self._sitk_image(i))

    @field
    def spacing(self, i):
        return self._sitk_image(i).GetSpacing()[::-1]

    @cached_property
    def _data(self):
        return pd.read_csv(self.root / 'LUNA25_Public_Training_Development_Data.csv')

    def _data_rows(self, i):
        return self._data[self._data['SeriesInstanceUID'] == i]

    def _data_column_value(self, i, column_name):
        values = self._data_rows(i).get(column_name).unique()
        assert len(values) == 1
        value = values[0]
        assert not pd.isnull(value)
        return value

    @field
    def patient_id(self, i):
        return str(self._data_column_value(i, 'PatientID'))

    @field
    def study_date(self, i):
        study_date = str(self._data_column_value(i, 'StudyDate'))
        return datetime.strptime(study_date, "%Y%m%d").date()

    @field
    def age(self, i):
        return self._data_column_value(i, 'Age_at_StudyDate')

    @field
    def gender(self, i):
        return self._data_column_value(i, 'Gender')

    @field
    def nodules(self, i):
        nodules = []
        sitk_image = self._sitk_image(i)
        shape = self.image(i).shape
        bbox_size = np.array([64, 128, 128])  # all nodule blocks in LUNA25 are of the same size
        for row in self._data_rows(i).itertuples():
            coords = (row.CoordX, row.CoordY, row.CoordZ)
            center_voxel = sitk_image.TransformPhysicalPointToIndex(map(int, coords))[::-1]

            nodule_block_origin = self.get_nodule_block_metadata(row.AnnotationID)['origin'][::-1]
            bbox_start_point = sitk_image.TransformPhysicalPointToIndex(map(int, nodule_block_origin))[::-1]
            bbox = np.array([bbox_start_point, np.minimum(bbox_start_point + bbox_size, shape)])
            nodules.append(
                LUNA25Nodule(
                    coords=coords,
                    lesion_id=row.LesionID,
                    annotation_id=str(row.AnnotationID),
                    nodule_id=str(row.NoduleID),
                    malignancy=row.label,
                    center_voxel=center_voxel,
                    bbox=bbox,
                )
            )
        return nodules

    def get_nodule_block_image(self, annotation_id):
        return np.load(self.root / f'luna25_nodule_blocks/image/{annotation_id}.npy')

    def get_nodule_block_metadata(self, annotation_id):
        metadata = np.load(self.root / f'luna25_nodule_blocks/metadata/{annotation_id}.npy', allow_pickle=True)
        assert metadata.shape == ()
        return metadata.item()
