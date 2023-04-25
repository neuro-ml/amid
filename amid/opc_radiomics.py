from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
from connectome import Source, meta, Output
from connectome.interface.nodes import Silent
from dicom_csv import (
    get_orientation_matrix,
    get_slice_locations,
    get_voxel_spacing,
    join_tree,
    stack_images,
)
from dicom_csv.rtstruct import contours_to_image
from pydicom import Dataset, dcmread

from .internals import checksum, licenses, register
from .utils import series_from_dicom_folder


@register(
    body_region='Head',
    license=licenses.CC_BYNCSA_40,
    link='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=33948764',
    modality=('MRI T1', 'MRI T1Gd', 'MRI T2', 'MRI T2-FLAIR'),
    prep_data_size='62G',
    raw_data_size='62G',
    task=('Segmentation', 'Classification'),
)
@checksum('opc_radiomcs')
class OPCRadiomics(Source):
    """
    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.
    version : str, optional
        the data version. Only has effect if the library was installed from a cloned git repository.

    Notes
    -----
    Download links:
    https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=33948764
    
    Please note:
    For some cases, there are HTV RTSTRUCT information listed, without GTV contours.
    HTV refers to a preoperative GTV target region, for instance in the setting of tonsillectomy.
    If there exist both GTV and HTV for a case, then there is probably a postoperative residual tumor seen.
    
    We return {'GTV': Union[np.ndarray, None], 'HTV': Union[np.ndarray, None]} as mask

    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = OPCReadiomics(root='/path/to/archives/root')
    >>> print(len(ds.ids))
    # 484
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 153)

    References
    ----------
    """

    _root: str = None

    def _base(_root: Silent) -> Path:
        if _root is None:
            raise ValueError("Please, provide path to folder containing the data")
        return Path(_root)
    
    @lru_cache(None)
    def _tree(_base) -> pd.DataFrame:
        if (_base / "opc.csv").exists():
            tree = pd.read_csv(_base / "opc.csv")
        else:
            tree = join_tree(_base, ignore_extensions=".csv")
        return tree[tree["FileName"] != "LICENSE"].set_index("PatientID")

    @meta
    def ids(_tree) -> List[str]:
        # 'OPC-00205' has no annotation
        return sorted(set(_tree.index))
    
    def _series(i, _tree, _base) -> List[Dataset]:
        series_uid = _tree.loc[i]["SeriesInstanceUID"].value_counts().index[0]
        return series_from_dicom_folder(_base / series_uid)
        
    def image(_series) -> np.ndarray:
        return stack_images(_series)
    
    def mask(i, _series, _tree, _base, tumor_tags: Output) -> Dict[str, Union[np.ndarray, None]]:
        """
        By default returns GTV (postoperative residual tumor) annotation.
        If there is none, returns HTV (preoperative tumor).
        """
        if len(_tree.loc[i]["SeriesInstanceUID"].value_counts().index.to_list()) < 2:
            return {"mask": None}
        
        rtstruct_path = _base / _tree.loc[i]["SeriesInstanceUID"].value_counts().index[1]
        contours = contours_to_image(_series, dcmread(rtstruct_path / "1-1.dcm"))
        mask = dict()
        for (name, label), contour in contours.items():
            if name in tumor_tags:
                mask[name] = contour.get_mask().astype(bool)
        return mask
        

    def tumor_tags(i, _series, _tree, _base) -> Tuple[str, ...]:
        if len(_tree.loc[i]["SeriesInstanceUID"].value_counts().index.to_list()) < 2:
            return tuple()
        
        tags = []
        
        rtstruct_path = _base / _tree.loc[i]["SeriesInstanceUID"].value_counts().index[1]
        contours = contours_to_image(_series, dcmread(rtstruct_path / "1-1.dcm"))
        for (name, label), contour in contours.items():
            if "gtv" in name.lower() or "htv" in name.lower():
                tags.append(name)
                
        return tuple(sorted(tags))
    
    def spacing(_series):
        return get_voxel_spacing(_series)
    
    def affine(_series):
        return get_orientation_matrix(_series)
    
    def slice_locations(_series):
        return get_slice_locations(_series)
