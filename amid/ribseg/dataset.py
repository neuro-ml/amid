from functools import lru_cache
from pathlib import Path

import nibabel
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent
from pandas import read_excel

from ..internals import checksum, licenses, register


@register(
    body_region='Chest',
    license=licenses.CC_BYNC_40,
    link='https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view?usp=sharing',
    modality='CT',
    raw_data_size='1.5 G',
    task='Segmentation',
)
@checksum('ribseg')
class RibSeg(Source):
    """
    Rib segmentation benchmark, named RibSeg, based on RibFrac dataset
    (https://ribfrac.grand-challenge.org/). Dataset contains 660 segmentaion
    masks (seg) and centerlines (cl).


    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.


    Notes
    -----
    Data downloaded from here:
    At the time of September 2023 the dataset was pre-launched, for details go to https://github.com/M3DV/RibSeg
    dataset: https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view?usp=sharing
    description table: https://docs.google.com/spreadsheets/d/1lz9liWPy8yHybKCdO3BCA9K76QH8a54XduiZS_9fK70/edit?usp=sharing

    seg: seg = nibabel.load(file name).get_fdata()
    # seg is a np array / volume of (512,512,N) with rib labels
    cl: cl = np.load(file name)['cl']
    # cl is a np array of (24,500,3), each rib contains 500 points

    References
    ----------
    Liang Jin, Shixuan Gu, Donglai Wei, Jason Ken Adhinarta,
    Kaiming Kuang, Yongjie Jessica Zhang, et al (2022)
    RibSeg v2: A Large-scale Benchmark for
    Rib Labeling and Anatomical Centerline Extraction
    """

    _root: str

    def _base(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the path to the root folder to the `root` argument')
        return Path(_root)

    @meta
    def ids(_base):
        result = [v.name.split('-')[0] for v in (_base / 'seg').iterdir()]
        return tuple(sorted(result))

    def seg(i, _base):
        seg_path = _base / 'seg' / f'{i}-rib-seg.nii.gz'
        ribs = nibabel.load(seg_path).get_fdata()
        return ribs.astype(np.int16)

    def cl(i, _base):
        cl_path = _base / 'cl' / f'{i}.npz'
        cl = np.load(cl_path)['cl']
        return cl.astype(np.int16)

    @lru_cache(None)
    def _meta(_base):
        file = 'RibSeg v2.xlsx'
        return read_excel(_base / file)
