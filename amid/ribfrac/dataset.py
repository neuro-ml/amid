from pathlib import Path

import nibabel
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import licenses, normalize


class RibFracBase(Source):
    """
    RibFrac dataset is a benchmark for developping algorithms on rib fracture detection,
    segmentation and classification. We hope this large-scale dataset could facilitate
    both clinical research for automatic rib fracture detection and diagnoses,
    and engineering research for 3D detection, segmentation and classification.


    Parameters
    ----------
    root : str, Path, optional
        path to the folder containing the raw downloaded archives.
        If not provided, the cache is assumed to be already populated.


    Notes
    -----
    Data downloaded from here:
    https://doi.org/10.5281/zenodo.3893507 -- train Part1 (300 images)
    https://doi.org/10.5281/zenodo.3893497 -- train Part2 (120 images)
    https://doi.org/10.5281/zenodo.3893495 -- val (80 images)
    https://zenodo.org/record/3993380 -- test (160 images without annotation)



    References
    ----------
    Jiancheng Yang, Liang Jin, Bingbing Ni, & Ming Li. (2020).
    RibFrac Dataset: A Benchmark for Rib Fracture Detection,
    Segmentation and Classification
    """

    _root: str

    def _base(_root: Silent):
        if _root is None:
            raise ValueError('Please pass the path to the root folder to the `root` argument')
        return Path(_root)

    @meta
    def ids(_base):
        result = set()
        for folder in ['Part1', 'Part2', 'ribfrac-val-images', 'ribfrac-test-images']:
            result |= {v.name.split('-')[0] for v in (_base / folder).iterdir()}

        return tuple(sorted(result))

    def _id2folder(_base):
        folders = [item for item in _base.iterdir() if item.is_dir()]
        result_dict = {}
        for folder in folders:
            p = _base / folder
            folder_ids = [v.name.split('-')[0] for v in p.iterdir()]
            folder_dict = {_id: p for _id in folder_ids}
            result_dict = {**result_dict, **folder_dict}

        return result_dict

    def image(i, _id2folder):
        image_path = _id2folder[i] / f'{i}-image.nii.gz'
        image = nibabel.load(image_path).get_fdata()
        return image.astype(np.int16)

    def label(i, _id2folder):
        folder_path = _id2folder[i]
        folder = folder_path.name
        if folder != 'ribfrac-test-images':
            if folder.startswith('Part'):
                label_path = folder_path / f'{i}-label.nii.gz'
            elif folder == 'ribfrac-val-images':
                dir = folder_path.parent / 'ribfrac-val-labels'
                label_path = dir / f'{i}-label.nii.gz'

            label = nibabel.load(label_path).get_fdata()
            return label.astype(np.int16)

    def affine(i, _id2folder):
        """The 4x4 matrix that gives the image's spatial orientation"""
        image_path = _id2folder[i] / f'{i}-image.nii.gz'
        return nibabel.load(image_path).affine


RibFrac = normalize(
    RibFracBase,
    'RibFrac',
    'ribfrac',
    body_region='Chest',
    license=licenses.CC_BYNC_40,
    link='https://ribfrac.grand-challenge.org',
    modality='CT',
    raw_data_size='77.8 G',
    task='Segmentation',
)
