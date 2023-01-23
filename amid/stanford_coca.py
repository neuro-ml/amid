import plistlib
import warnings
from functools import lru_cache

import pandas as pd
import pydicom
from ast import literal_eval
from enum import IntEnum
from pathlib import Path

import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent, Output
from dicom_csv import expand_volumetric, drop_duplicated_instances, drop_duplicated_slices, order_series, stack_images, \
    get_pixel_spacing, get_slice_locations, get_orientation_matrix
from skimage.draw import polygon

from .internals import checksum, licenses, register


class CoCaClasses(IntEnum):
    LAD = 1
    LCX = 2
    RCA = 3
    LCA = 4


@register(
    body_region=('Coronary', 'Chest'),
    license=licenses.CC_BYNCND_40,
    link='https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa',
    modality='CT',
    prep_data_size=None,  # TODO: should be measured...
    raw_data_size='28G',
    task='Segmentation',
)
@checksum('stanford_coca')
class StanfordCoCa(Source):
    """
    A Stanford AIMI's Co(ronary) Ca(lcium) dataset.


    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Follow the download instructions at https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa.
    You'll need to register and accept the terms of use. After that, copy the files from Azure:

    azcopy copy 'some-generated-access-link' /path/to/downloaded/data/ --recursive=true

    Then, the folder with raw downloaded data should contain two subfolders - a subset with gated coronary CT scans
    and corresponding coronary calcium segmentation masks (`Gated_release_final`)
    and a folder with the non-gated CT scans with corresponding coronary with coronary artery calcium scores
    (`deidentified_nongated`).

    The folder with gated data should have original structure:
        ./Gated_release_final/patient/0/folder-with-dcms/
        ./Gated_release_final/calcium_xml/0.xml
        ...

    The folder with nongated data should have original structure:
        ./deidentified_nongated/0/folder-with-dcms/
        ...

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = StanfordCoCa(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 787  # actually dunno
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 163)
    >>> print(ds.mask(ds.ids[80]).shape)
    # (512, 512, 771)

    """

    _root: str = None
    _raise: bool = False
    _class_abbr: dict = {
        'Left Anterior Descending Artery': 'LAD',
        'Left Circumflex Artery': 'LCA',
        'Right Coronary Artery': 'RCA',
        'Left Coronary Artery': 'LCA',
    }

    def _split(i):
        return i.split('-')[0]

    def _i(i):
        return i.split('-')[1]

    def _folder_with_images(_split):
        if _split == 'gated':
            return Path('Gated_release_final')/'patient'
        if _split == 'nongated':
            return 'deidentified_nongated'
        raise ValueError("Unknown split. Use 'gated' or 'nongated' options.")

    def _folder_with_annotations(_split):
        if _split == 'gated':
            return Path('Gated_release_final') / 'calcium_xml'
        if _split == 'nongated':
            return None
        raise ValueError("Unknown split. Use 'gated' or 'nongated' options.")

    @meta
    def ids(_root: Silent):
        gated_ids = tuple(sorted('gated-'+x.name for x in (Path(_root) / 'Gated_release_final'/'patient').iterdir() if x.is_dir()))
        nongated_ids = tuple(sorted('nongated-'+x.name for x in (Path(_root) / 'deidentified_nongated').iterdir() if x.is_dir()))
        return gated_ids + nongated_ids

    def _series(_i, _root: Silent, _folder_with_images):
        folder_with_dicoms  = Path(_root) / _folder_with_images / _i
        series = list(map(pydicom.dcmread, folder_with_dicoms.glob('*/*.dcm')))
        # series = sorted(series, key=lambda x: x.InstanceNumber)
        series = expand_volumetric(series)
        series = drop_duplicated_instances(series)

        if True:  # drop_dupl_slices
            _original_num_slices = len(series)
            series = drop_duplicated_slices(series)
            if len(series) < _original_num_slices:
                warnings.warn(f'Dropped duplicated slices for series {series[0]["StudyInstanceUID"]}.')

        series = order_series(series, decreasing=False)
        return series

    def image(i, _series):
        image = stack_images(_series, -1).astype(np.int16)
        return image

    def _image_meta(_series):
        metas = [list(dict(s).values()) for s in _series]
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

    def image_meta(_image_meta):
        return _image_meta

    def study_uid(_series):
        study_uids = np.unique([x["StudyInstanceUID"] for x in _series])
        assert len(study_uids) == 1
        # series_id_to_study
        return study_uids[0]

    def pixel_spacing(_series):
        return get_pixel_spacing(_series).tolist()

    def slice_locations(_series):
        return get_slice_locations(_series)

    def orientation_matrix(_series):
        return get_orientation_matrix(_series)

    def mask(_i, image: Output, slice_locations: Output, _image_meta,
             _root: Silent, _folder_with_annotations, _class_abbr, _raise):
        if _folder_with_annotations is None:
            warnings.warn("The used split doesn't contain segmentation masks.")
            return None

        try:
            with open(Path(_root)/_folder_with_annotations/f'{_i}.xml', 'rb') as fp:
                annotation = plistlib.load(fp)
                image_annotations = annotation['Images']

        except FileNotFoundError as e:
            if _raise:
                raise e
            else:
                warnings.warn(f"Missing annotation for id: {_i}")
                return None

        shape = image.shape
        sl = slice_locations
        multiclass_mask = np.zeros(shape, np.uint8)
        try:
            for slice_annotation in image_annotations:
                for roi in slice_annotation['ROIs']:
                    if roi['Area'] > 0:
                        assert roi['Name'] in _class_abbr, f"Unexpected class: {roi['Name']}"
                        class_name = _class_abbr[roi['Name']]
                        class_id = CoCaClasses[class_name].value
                        slice_location = literal_eval(roi['Point_mm'][0])[-1]
                        slice_id = np.argwhere(sl == slice_location).squeeze()
                        assert slice_id, f"Slice where calcification is located is not presented in the series." #(i, slice_location, sl)
                        roi_contour = [literal_eval(x) for x in roi['Point_px']]
                        slice_mask = np.zeros(shape[:2])
                        xs, ys = polygon(*(np.array(roi_contour).T), shape[:2])
                        slice_mask[ys, xs] = True
                        multiclass_mask[..., slice_id] = (class_id * slice_mask.astype(int)).astype(np.uint8)

        except AssertionError as e:
            if _raise:
                raise e
            else:
                warnings.warn(f"Mask preparation for idx {_i} failed with: '{str(e)}'. Returning None")
                return None

        return multiclass_mask

    @lru_cache(None)
    def _scores(_root: Silent, _folder_with_images):
        p = Path(_root) / _folder_with_images / 'scores.xlsx'

        if not p.exists():
            return None

        return pd.read_excel(p, index_col=0)

    def score(_i, _scores):
        if _scores is None:
            return None
        return _scores.loc[_i+'A'].to_dict()
