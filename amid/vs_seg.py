import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from connectome import Source, meta
from connectome.interface.nodes import Output, Silent
from dicom_csv import (
    get_common_tag,
    get_orientation_matrix,
    get_pixel_spacing,
    get_slice_locations,
    get_tag,
    join_tree,
    order_series,
    stack_images,
)
from dicom_csv.exceptions import ConsistencyError, TagTypeError
from scipy import stats
from skimage.draw import polygon

from amid.internals import checksum, register


@register(
    body_region='Head',
    license='CC BY 4.0',
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053',
    modality=('MRI T1c', 'MRI T2'),
    prep_data_size=None,  # TODO: should be measured...
    raw_data_size='27G',
    task='Segmentation',
)
@checksum('vs_seg')
class VSSEG(Source):
    """
    Segmentation of vestibular schwannoma from MRI, an open annotated dataset ... (VS-SEG) [1]_.

    The dataset contains 250 pairs of T1c and T2 images of the brain with the
    vestibular schwannoma segmentation task.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    The dataset and corresponding metadata could be downloaded at the TCIA page:
    https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053.

    To download DICOM images using `.tcia` file, we use public build of TCIA downloader:
    https://github.com/ygidtu/NBIA_data_retriever_CLI.

    Then, download the rest of metadata from TCIA page:
      - `DirectoryNamesMappingModality.csv`
      - `Vestibular-Schwannoma-SEG_matrices Mar 2021.zip`
      - `Vestibular-Schwannoma-SEG contours Mar 2021.zip`

    and unzip the latter two `.zip` archives.

    So the root folder to pass to this dataset should contain 3 folders and 1 `.csv` file:
        <...>/DirectoryNamesMappingModality.csv
        <...>/Vestibular-Schwannoma-SEG/
                ├── VS-SEG-001/...
                ├── VS-SEG-002/...
                └── ...
        <...>/contours/
        <...>/registration_matrices/

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = VSSEG(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 250
    >>> print(ds.image_t1(ds.ids[0]).shape)
    # (512, 512, 120)
    >>> print(ds.schwannoma_t1(ds.ids[80]).shape)
    # (512, 512, 120)

    References
    ----------
    .. [1] Shapey, Jonathan, et al. "Segmentation of vestibular schwannoma from MRI,
           an open annotated dataset and baseline algorithm."
           Scientific Data 8.1 (2021): 1-6.
           https://www.nature.com/articles/s41597-021-01064-w

    """

    _root: str = None

    @meta
    def ids(_root):
        return tuple(sorted(os.listdir(Path(_root) / 'Vestibular-Schwannoma-SEG')))

    def registration_matrix_t1_to_t2_params(i, _root: Silent):
        for file in (Path(_root) / 'registration_matrices').glob(f'vs_gk_{int(i.split("-")[-1])}_t1/*'):
            return sitk.ReadTransform(str(file)).GetParameters()

    def registration_matrix_t2_to_t1_params(i, _root: Silent):
        for file in (Path(_root) / 'registration_matrices').glob(f'vs_gk_{int(i.split("-")[-1])}_t2/*'):
            return sitk.ReadTransform(str(file)).GetParameters()

    # ### contours and masks ###

    def contours_t1(i, _root: Silent):
        for file in (Path(_root) / 'contours').glob(f'vs_gk_{int(i.split("-")[-1])}_t1/*'):
            with open(str(file), 'r') as f:
                return json.load(f)

    def contours_t2(i, _root: Silent):
        for file in (Path(_root) / 'contours').glob(f'vs_gk_{int(i.split("-")[-1])}_t2/*'):
            with open(str(file), 'r') as f:
                return json.load(f)

    def normed_contours_t1(contours_t1: Output, voxel_spacing_t1: Output, patient_position_t1: Output):
        return _norm_contours(contours_t1, voxel_spacing_t1, patient_position_t1)

    def normed_contours_t2(contours_t2: Output, voxel_spacing_t2: Output, patient_position_t2: Output):
        return _norm_contours(contours_t2, voxel_spacing_t2, patient_position_t2)

    def has_schwannoma(contours_t1: Output):
        return _get_schwannoma_structure_name(contours_t1) is not None

    def has_cochlea(contours_t1: Output):
        return _get_cochlea_structure_name(contours_t1) is not None

    def has_meningioma(contours_t1: Output):
        return _get_meningioma_structure_name(contours_t1) is not None

    def schwannoma_t1(normed_contours_t1: Output, image_t1: Output):
        return _get_mask(normed_contours_t1, image_t1.shape, obj='schwannoma')

    def schwannoma_t2(normed_contours_t2: Output, image_t2: Output):
        return _get_mask(normed_contours_t2, image_t2.shape, obj='schwannoma')

    def cochlea_t1(normed_contours_t1: Output, image_t1: Output):
        return _get_mask(normed_contours_t1, image_t1.shape, obj='cochlea')

    def cochlea_t2(normed_contours_t2: Output, image_t2: Output):
        return _get_mask(normed_contours_t2, image_t2.shape, obj='cochlea')

    def meningioma_t1(normed_contours_t1: Output, image_t1: Output):
        return _get_mask(normed_contours_t1, image_t1.shape, obj='meningioma')

    def meningioma_t2(normed_contours_t2: Output, image_t2: Output):
        return _get_mask(normed_contours_t2, image_t2.shape, obj='meningioma')

    # ### series and images: ###

    def _series_t1(i, _root: Silent):
        return _load_series(i, _root, 'T1 image')

    def _series_t2(i, _root: Silent):
        return _load_series(i, _root, 'T2 image')

    def image_t1(_series_t1):
        return stack_images(_series_t1, -1).transpose(1, 0, 2)

    def image_t2(_series_t2):
        return stack_images(_series_t2, -1).transpose(1, 0, 2)

    # ### metadata: ###

    def study_uid(_series_t1):
        return get_common_tag(_series_t1, 'StudyInstanceUID')

    def series_uid_t1(_series_t1):
        return get_common_tag(_series_t1, 'SeriesInstanceUID')

    def series_uid_t2(_series_t2):
        return get_common_tag(_series_t2, 'SeriesInstanceUID')

    def pixel_spacing_t1(_series_t1):
        return get_pixel_spacing(_series_t1).tolist()

    def pixel_spacing_t2(_series_t2):
        return get_pixel_spacing(_series_t2).tolist()

    def slice_locations_t1(_series_t1):
        return get_slice_locations(_series_t1)

    def slice_locations_t2(_series_t2):
        return get_slice_locations(_series_t2)

    def voxel_spacing_t1(pixel_spacing_t1: Output, slice_locations_t1: Output):
        return (*pixel_spacing_t1, stats.mode(np.diff(slice_locations_t1))[0].item())

    def voxel_spacing_t2(pixel_spacing_t2: Output, slice_locations_t2: Output):
        return (*pixel_spacing_t2, stats.mode(np.diff(slice_locations_t2))[0].item())

    def orientation_matrix_t1(_series_t1):
        return get_orientation_matrix(_series_t1)

    def orientation_matrix_t2(_series_t2):
        return get_orientation_matrix(_series_t2)

    def patient_position_t1(_series_t1):
        return tuple(map(float, get_tag(_series_t1[0], 'ImagePositionPatient')))

    def patient_position_t2(_series_t2):
        return tuple(map(float, get_tag(_series_t2[0], 'ImagePositionPatient')))

    def patient_id(_series_t1):
        return get_common_tag(_series_t1, 'PatientID', default=None)

    def study_date(_series_t1):
        return _get_study_date(_series_t1)


def _load_series(_id, root, modality):
    root = Path(root)

    df = pd.read_csv(root / 'DirectoryNamesMappingModality.csv')
    df = df[df['Classic Directory Name'].apply(lambda x: _id in x)]

    study_id, series_id = df[df['Modality'] == modality]['Classic Directory Name'].iloc[0].split('/')[1:]
    some_date = os.listdir(root / 'Vestibular-Schwannoma-SEG' / _id / study_id)[0]
    path_to_series = root / 'Vestibular-Schwannoma-SEG' / _id / study_id / some_date / series_id
    tree = join_tree(path_to_series)
    tree = tree[tree['NoError']]

    series = [pydicom.dcmread(path_to_series / fname) for fname in tree['FileName'].values]
    series = order_series(series, decreasing=False)
    return series


def _norm_contours(contours, voxel_spacing, origin):
    voxel_spacing = np.float32(voxel_spacing)
    pixel_spacing, z_spacing = voxel_spacing[:-1], voxel_spacing[-1]
    origin = np.float32(origin)

    contours_normed = []

    for contour in contours:
        cs = contour['LPS_contour_points']
        cs_normed = []
        for c in cs:
            c = np.float32(c)
            xy_normed = (c[:, :-1] - origin[:-1]) / pixel_spacing
            z_normed = np.round((c[0][-1] - origin[-1]) / z_spacing)
            c_normed = np.concatenate((xy_normed, np.repeat([[z_normed]], xy_normed.shape[0], axis=0)), axis=-1)
            cs_normed.append(c_normed.tolist())
        contours_normed.append({**contour, 'LPS_contour_points': cs_normed})

    return contours_normed


def _get_study_date(series):
    # FIXME: this is the duplicated code from cancer_500
    try:
        study_date = get_common_tag(series, 'StudyDate')
    except (TagTypeError, ConsistencyError):
        return

    if not isinstance(study_date, str) or not study_date.isnumeric() or len(study_date) != 8:
        return

    try:
        year = int(study_date[:4])
        month = int(study_date[4:6])
        day = int(study_date[6:])
    except TypeError:
        return

    if year < 1972:  # year of creation of first CT scanner
        return

    return datetime.date(year, month, day)


def _contour2mask(cnt: list, shape):
    msk = np.zeros(shape, dtype=np.uint8)
    for planar_c in cnt:
        idx = int(planar_c[0][-1])
        slc = msk[..., idx]
        r, c, _ = np.transpose(planar_c)
        rr, cc = polygon(r, c, shape=slc.shape)
        slc[rr, cc] = 1
    return msk


def _contours2names(contours: list):
    return [c['structure_name'] for c in contours]


def _select_contour_by_structure_name(contours: list, name: str):
    return [c for c in contours if c['structure_name'] == name][0]['LPS_contour_points']


def _get_schwannoma_structure_name(contours: list):
    names = _contours2names(contours)

    # filter 1:
    filter_names = (
        'Brainstem',
        'Modiolus',  # CS (other)
        'Cochlea',
        'Cochlea_c',
        'Cochlea_d',
        'cochlea',  # CS (Cochlea)
        'Test',  # a duplicate of 'TV' with Dice Score = 0.92 (seems to be a worse contour)
        'Men',  # is a meningioma case
    )
    names = [name for name in names if name not in filter_names]

    if len(names) == 0:
        return None

    if len(names) == 1:
        if names[0] == 'vol 2y':
            # 'VS-SEG-063' -- is a meningioma case + "vol 2y" contour has a bad interpolation:(
            return None
        else:
            return names[0]

    # filter 2:
    # no 'TV' simultaneously with 'AN'
    higher_priority_names = (
        'TV',
        'tv',
        'AN',
    )
    is_high_priority = [name in higher_priority_names for name in names]
    if np.sum(is_high_priority) == 1:
        return np.array(names)[np.array(is_high_priority)][0]

    return names


def _get_cochlea_structure_name(contours: list):
    names = _contours2names(contours)
    cochlea_names = ('Cochlea', 'Cochlea_c', 'Cochlea_d', 'cochlea')
    # there are no duplicated cochlea structures:
    for name in names:
        if name in cochlea_names:
            return name


def _get_meningioma_structure_name(contours: list):
    return 'Men' if 'Men' in _contours2names(contours) else None


def _get_mask(contours: list, shape, obj: str):
    get_structure_name = {
        'schwannoma': _get_schwannoma_structure_name,
        'cochlea': _get_cochlea_structure_name,
        'meningioma': _get_meningioma_structure_name,
    }[obj]
    structure_name = get_structure_name(contours)
    if structure_name is not None:
        contour = _select_contour_by_structure_name(contours, structure_name)
        return _contour2mask(contour, shape)
