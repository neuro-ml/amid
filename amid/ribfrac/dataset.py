from pathlib import Path

import nibabel
import numpy as np
from connectome import Output, Source, meta
from connectome.interface.nodes import Silent
from dpipe.io import load

from ..internals import checksum, licenses, register
from ..utils import deprecate


@register(
    body_region='Chest',
    license=licenses.CC_BYNC_40,
    link='https://ribfrac.grand-challenge.org',
    modality='CT',
    raw_data_size='77.8 G',
    task='Segmentation',
)
@checksum('ribfrac')
class RibFrac(Source):
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
    def ids(_root: Silent):
        result = set()
        for folder in ['Part1', 'Part2', 'ribfrac-val-images', 'ribfrac-test-images']:
            result |= {v.name.split('-')[0] for v in (Path(_root) / folder).iterdir()}

        return tuple(sorted(result))

    def _id2folder(_root: Silent):
        folders = [item for item in Path(_root).iterdir() if item.is_dir()]
        result_dict = {}
        for folder in folders:
            p = _root / folder
            folder_ids = [v.name.split('-')[0] for v in p.iterdir()]
            folder_dict = {_id: p for _id in folder_ids}
            result_dict = {**result_dict, **folder_dict}

        return result_dict

    def image(i, _id2folder):
        image_path = _id2folder[i] / f'{i}-image.nii.gz'
        image = load(image_path)
        image = np.swapaxes(image, 0, 1)[:, :, ::-1]
        return image.astype(np.int16)

    def label(i, _id2folder):
        folder_path = Path(_id2folder[i])
        folder = folder_path.name
        if folder != 'ribfrac-test-images':
            if folder.startswith('Part'):
                label_path = folder_path / f'{i}-label.nii.gz'
                label = load(label_path)
            elif folder == 'ribfrac-val-images':
                dir = folder_path.parent / 'ribfrac-val-labels'
                label = load(dir / f'{i}-label.nii.gz')

            label = np.swapaxes(label, 0, 1)[:, :, ::-1]
            return label.astype(np.int16)

    def spacing(i, _id2folder):
        """Returns voxel spacing along axes (x, y, z)."""
        nii_path = Path(_id2folder[i]) / f'{i}-image.nii.gz'
        nii_header = nibabel.load(str(nii_path)).header
        spacing = nii_header.get_zooms()
        assert spacing[0] == spacing[1]  # important as we swap axes
        return spacing

    @deprecate(message='Use `spacing` method instead.')
    def voxel_spacing(spacing: Output):
        return spacing
