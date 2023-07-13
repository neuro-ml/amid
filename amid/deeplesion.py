# TODO: Add annotation info from DL_info.csv

from functools import lru_cache
from pathlib import Path

import deli
import nibabel
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from .internals import checksum, register


@register(
    body_region=('Abdomen', 'Thorax'),
    license='DeepLesion data license',  # TODO
    link='https://nihcc.app.box.com/v/DeepLesion',
    modality="CT",
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
    Once you download the data and unarchive all 56 zip archives, you should run `DL_save_nifti.py` provided by the authors
    to convert 2D PNGs into 20094 nii.gz files.

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
        return tuple(sorted([file.name.replace(".nii.gz", "") for file in (_base / "Images_nifti").glob("*.nii.gz")]))

    def _image_file(i, _base):
        return nibabel.load(_base / "Images_nifti" / f"{i}.nii.gz")

    @lru_cache
    def _metadata(_base):
        return deli.load(_base / "DL_info.csv")

    def _row(i, _metadata):
        patient, study, series = map(int, i.split("_")[:3])
        return _metadata.query("Patient_index==@patient").query("Study_index==@study").query("Series_ID==@series")

    def patient_id(i):
        patient, study, series = map(int, i.split("_")[:3])
        return patient

    def study_id(i):
        patient, study, series = map(int, i.split("_")[:3])
        return study

    def series_id(i):
        patient, study, series = map(int, i.split("_")[:3])
        return series

    def sex(_row):
        return _row.Patient_gender.iloc[0]

    def age(_row):
        """Patient Age might be different for different studies (dataset contains longitudinal records)."""
        return _row.Patient_age.iloc[0]

    def ct_window(_row):
        return _row.DICOM_windows.iloc[0]

    def affine(_image_file):
        return _image_file.affine

    def spacing(_image_file):
        return tuple(_image_file.header['pixdim'][1:4])

    def image(_image_file):
        return np.asarray(_image_file.dataobj)

    def train_val_fold(_row):
        return int(_row.Train_Val_Test.iloc[0])
