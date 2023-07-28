from functools import lru_cache
from pathlib import Path

import deli
import nibabel
import numpy as np
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent

from .internals import checksum, register


@register(
    body_region=('Abdomen', 'Thorax'),
    license='DeepLesion data license',  # TODO
    link='https://nihcc.app.box.com/v/DeepLesion',
    modality='CT',
    prep_data_size='259G',
    raw_data_size='259G',
    task=('Localisation', 'Detection', 'Classification'),
)
@checksum('deeplesion')
class DeepLesion(Source):
    """
    DeepLesion is composed of 33,688 bookmarked radiology images from
    10,825 studies of 4,477 unique patients. For every bookmarked image, a bound-
    ing box is created to cover the target lesion based on its measured diameters [1].

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing `DL_info.csv` file and a subfolder `Images_nifti` with 20094 nii.gz files.

    Notes
    -----
    Dataset is available at https://nihcc.app.box.com/v/DeepLesion

    To download the data we recommend using a Python script provided by the authors `batch_download_zips.py`.
    Once you download the data and unarchive all 56 zip archives, you should run `DL_save_nifti.py`
    provided by the authors to convert 2D PNGs into 20094 nii.gz files.

    Example
    --------
    >>> ds = DeepLesion(root='/path/to/folder')
    >>> print(len(ds.ids))
    # 20094

    References
    ----------
    .. [1] Yan, Ke, Xiaosong Wang, Le Lu, and Ronald M. Summers. "Deeplesion: Automated deep mining,
      categorization and detection of significant radiology image findings using large-scale clinical
      lesion annotations." arXiv preprint arXiv:1710.01766 (2017).

    """

    _root: str = None

    def _base(_root: Silent):
        if _root is None:
            raise ValueError('Please provide the `root` argument')
        return Path(_root)

    @meta
    def ids(_base):
        return tuple(sorted([file.name.replace('.nii.gz', '') for file in (_base / 'Images_nifti').glob('*.nii.gz')]))

    def _image_file(i, _base):
        return nibabel.load(_base / 'Images_nifti' / f'{i}.nii.gz')

    @lru_cache(None)
    def _metadata(_base):
        df = deli.load(_base / 'DL_info.csv')

        cols_to_transform = [
            'Measurement_coordinates',
            'Bounding_boxes',
            'Lesion_diameters_Pixel_',
            'Normalized_lesion_location',
        ]
        for col in cols_to_transform:
            df[col] = df[col].apply(lambda x: list(map(float, x.split(','))))

        df['Slice_range_list'] = df['Slice_range'].apply(lambda x: list(map(int, x.split(','))))

        def get_ids(x):
            patient_study_series = '_'.join(x.File_name.split('_')[:3])
            slice_range_list = list(map(str, x.Slice_range_list))
            slice_range_list = [num.zfill(3) for num in slice_range_list]
            slice_range_list = '-'.join(slice_range_list)
            return f'{patient_study_series}_{slice_range_list}'

        df['ids'] = df.apply(get_ids, axis=1)
        return df

    def _row(i, _metadata):
        # funny story, f-string does not work for pandas.query,
        # @ syntax does not work for linter, use # noqa
        return _metadata.query('ids==@i')

    def patient_id(i):
        patient, study, series = map(int, i.split('_')[:3])
        return patient

    def study_id(i):
        patient, study, series = map(int, i.split('_')[:3])
        return study

    def series_id(i):
        patient, study, series = map(int, i.split('_')[:3])
        return series

    def sex(_row):
        return _row.Patient_gender.iloc[0]

    def age(_row):
        """Patient Age might be different for different studies (dataset contains longitudinal records)."""
        return _row.Patient_age.iloc[0]

    def ct_window(_row):
        """CT window extracted from DICOMs. Recall, that it is min-max values for windowing, not width-level."""
        return _row.DICOM_windows.iloc[0]

    def affine(_image_file):
        return _image_file.affine

    def spacing(_image_file):
        return tuple(_image_file.header['pixdim'][1:4])

    def image(_image_file):
        """Some 3D volumes are stored as separate subvolumes, e.g. ds.ids[15000] and ds.ids[15001]."""
        return np.asarray(_image_file.dataobj)

    def train_val_test(_row):
        """Authors' defined randomly generated patient-level data split, train=1, validation=2, test=3,
        70/15/15 ratio."""
        return int(_row.Train_Val_Test.iloc[0])

    def lesion_position(_row):
        """Lesion measurements as it appear in DL_info.csv, for details see
        https://nihcc.app.box.com/v/DeepLesion/file/306056134060 ."""
        position = _row[
            [
                'Slice_range_list',
                'Key_slice_index',
                'Measurement_coordinates',
                'Bounding_boxes',
                'Lesion_diameters_Pixel_',
                'Normalized_lesion_location',
            ]
        ].to_dict('list')
        position['Slice_range_list'] = position['Slice_range_list'][0]
        return position

    def mask(image: Output, lesion_position: Output):
        """Mask of provided bounding boxes. Recall that bboxes annotation
        is very coarse, it only covers a single 2D slice."""
        mask = np.zeros_like(image)
        min_index = lesion_position['Slice_range_list'][0]
        for i, slice_index in enumerate(lesion_position['Key_slice_index']):
            image_index = slice_index - min_index
            top_left_x, top_left_y, bot_right_x, bot_right_y = lesion_position['Bounding_boxes'][i]
            mask[
                int(np.floor(top_left_y)) : int(np.ceil(bot_right_y)),
                int(np.floor(top_left_x)) : int(np.ceil(bot_right_x)),
                image_index,
            ] = 1
        return mask
