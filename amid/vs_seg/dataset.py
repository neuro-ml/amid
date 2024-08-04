import json

import numpy as np
import pandas as pd
import pydicom
from dicom_csv import (
    get_common_tag,
    get_pixel_spacing,
    get_slice_locations,
    get_tag,
    join_tree,
    order_series,
    stack_images,
)
from scipy import stats
from skimage.draw import polygon

from ..internals import Dataset, licenses, register
from ..utils import get_series_date


@register(
    body_region='Head',
    license=licenses.CC_BY_40,
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053',
    modality=('MRI T1c', 'MRI T2'),
    prep_data_size='14,42G',
    raw_data_size='27G',
    task='Segmentation',
)
class VSSEG(Dataset):
    """
    Segmentation of vestibular schwannoma from MRI, an open annotated dataset ... (VS-SEG) [1]_.

    The dataset contains 250 pairs of T1c and T2 images of the brain with the
    vestibular schwannoma segmentation task.

    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.


    Notes
    -----
    The dataset and corresponding metadata could be downloaded at the TCIA page:
    https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053.

    To download DICOM images using `.tcia` file, we used public build of TCIA downloader:
    https://github.com/ygidtu/NBIA_data_retriever_CLI.

    Then, download the rest of metadata from TCIA page:
      - `DirectoryNamesMappingModality.csv`
      - `Vestibular-Schwannoma-SEG_matrices Mar 2021.zip`
      - `Vestibular-Schwannoma-SEG contours Mar 2021.zip`

    and unzip the latter two `.zip` archives.

    So the `root` folder should contain 3 folders and 1 `.csv` file:
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
    # 484
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 120)
    >>> print(ds.schwannoma(ds.ids[1]).shape)
    # (384, 384, 80)

    References
    ----------
    .. [1] Shapey, Jonathan, et al. "Segmentation of vestibular schwannoma from MRI,
           an open annotated dataset and baseline algorithm."
           Scientific Data 8.1 (2021): 1-6.
           https://www.nature.com/articles/s41597-021-01064-w

    """

    @property
    def ids(self):
        subject_id_paths = list((self.root / 'Vestibular-Schwannoma-SEG').glob('VS-SEG-*'))
        t1_ids = [p.name + '-T1' for p in subject_id_paths]
        t2_ids = [p.name + '-T2' for p in subject_id_paths]
        return tuple(sorted(t1_ids + t2_ids))

    def modality(self, i):
        return i.rsplit('-', 1)[1]

    def subject_id(self, i):
        return i.rsplit('-', 1)[0]

    # ### series and images: ###

    def _series(self, i):
        return _load_series(i, self.root)

    def image(self, i):
        return stack_images(self._series(i), -1).transpose(1, 0, 2)

    # ### contours and masks ###

    def _contours(self, i):
        subject_num = int(self.subject_id(i).rsplit('-', 1)[-1])
        for file in (self.root / 'contours').glob(f'vs_gk_{subject_num}_{self.modality(i).lower()}/*'):
            with open(str(file), 'r') as f:
                return json.load(f)

    def _pixel_spacing(self, i):
        return get_pixel_spacing(self, i).tolist()

    def _slice_locations(self, i):
        return get_slice_locations(self, i)

    def spacing(self, i):
        """The maximum relative difference in `slice_locations` < 1e-12,
        so we allow ourselves to use the common spacing for the whole 3D image."""
        return (*self._pixel_spacing(i), stats.mode(np.diff(self._slice_locations(i)))[0].item())

    def _normed_contours(self, i):
        return _norm_contours(self._contours(i), self.spacing(i), self._patient_position(i))

    def _patient_position(self, i):
        return tuple(map(float, get_tag(self._series(i)[0], 'ImagePositionPatient')))

    def schwannoma(self, i):
        return _get_mask(self._normed_contours(i), self.image(i).shape, obj='schwannoma')

    def cochlea(self, i):
        return _get_mask(self._normed_contours(i), self.image(i).shape, obj='cochlea')

    def meningioma(self, i):
        return _get_mask(self._normed_contours(i), self.image(i).shape, obj='meningioma')

    # ### other DICOM metadata: ###

    def study_uid(self, i):
        return get_common_tag(self._series(i), 'StudyInstanceUID')

    def series_uid(self, i):
        return get_common_tag(self._series(i), 'SeriesInstanceUID')

    def patient_id(self, i):
        return get_common_tag(self._series(i), 'PatientID', default=None)

    def study_date(self, i):
        return get_series_date(self, i)


def _load_series(_id, root):
    _id, modality = _id.rsplit('-', 1)
    df = pd.read_csv(root / 'DirectoryNamesMappingModality.csv')
    df = df[df['Classic Directory Name'].apply(lambda x: _id in x)]

    study_id, series_id = df[df['Modality'] == f'{modality} image']['Classic Directory Name'].iloc[0].split('/')[1:]
    next_single_folder = list((root / 'Vestibular-Schwannoma-SEG' / _id / study_id).glob('*'))[0].name
    path_to_series = root / 'Vestibular-Schwannoma-SEG' / _id / study_id / next_single_folder / series_id
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


def _contour2mask(cnt: list, shape):
    msk = np.zeros(shape, dtype=bool)
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
