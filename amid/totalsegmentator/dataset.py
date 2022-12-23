import os
import json
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import checksum, register


@register(
    body_region=("Head", "Thorax", "Abdomen", "Pelvis", "Legs"),
    license="CC BY 4.0",
    link="https://zenodo.org/record/6802614#.Y6M2MxXP1D8",
    modality="CT",
    prep_data_size=None,
    raw_data_size="35G",
    task="Supervised anatomical structures segmentation",
)
@checksum("totalsegmentator")
class Totalsegmentator(Source):
    """
    In 1204 CT images we segmented 104 anatomical structures (27 organs, 59 bones, 10 muscles, 8 vessels)
    covering a majority of relevant classes for most use cases.

    The CT images were randomly sampled from clinical routine, thus representing a real world dataset which generalizes to clinical application.

    The dataset contains a wide range of different pathologies, scanners, sequences and institutions. [1]

    Parameters
    ----------
    root : str, Path, optional
        path to the unpacked downloaded archive.
        If not provided, the cache is assumed to be already populated.

    Notes
    -----
    Download link: https://zenodo.org/record/6802614#.Y6M2MxXP1D8

    Examples
    --------
    >>> # Unpack the downloaded archive at any folder and pass the path to the constructor:
    >>> ds = FLARE2022(root="/path/to/unpacked/downloaded/archive")
    >>> print(len(ds.ids))
    # 1203
    >>> print(ds.image(ds.ids[0]).shape)
    # (294, 192, 179)
    >>> print(ds.aorta(ds.ids[25]).shape)
    # (320, 320, 145)

    References
    ----------
    .. [1] Jakob Wasserthal (2022) Dataset with segmentations of 104 important anatomical structures in 1204 CT images. Available at:
    https://zenodo.org/record/6802614#.Y6M2MxXP1D8
    """

    _root: str = None

    def _add_masks(scope):
        with open(Path(os.path.abspath(__file__)).parent / "anatomical_structures.json", "r") as f:
            anatomical_structures = json.load(f)

        def make_loader(anatomical_structure):
            def loader(i, _root: Silent):
                nii = Path(_root) / i / "segmentations" / f"{anatomical_structure}.nii.gz"
                return np.asarray(nibabel.load(nii).dataobj)

            return loader

        for anatomical_structure in anatomical_structures:
            scope[anatomical_structure] = make_loader(anatomical_structure)

    _add_masks(locals())

    @meta
    def ids(_root: Silent):
        if _root is None:
            raise ValueError("Please pass the location of the unpacked downloaded archive.")

        result = set([x.name for x in Path(_root).glob("*") if x.name != "meta.csv"])

        return sorted(result)

    def meta(_root: Silent):
        return pd.read_csv(Path(_root) / "meta.csv", sep=";")

    def image(i, _root: Silent):
        nii = Path(_root) / i / "ct.nii.gz"

        return np.asarray(nibabel.load(nii).dataobj)

    def affine(i, _root: Silent):
        """The 4x4 matrix that gives the image"s spatial orientation"""
        nii = Path(_root) / i / "ct.nii.gz"

        return nibabel.load(nii).affine
