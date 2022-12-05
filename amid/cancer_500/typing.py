from enum import Enum
from typing import NamedTuple, Optional, Sequence


class Texture(Enum):
    Solid, PartSolid, GroundGlass, Other = 0, 1, 2, 3


class Review(Enum):
    Confirmed, ConfirmedPartially, Doubt, Rejected = 0, 1, 2, 3


class Comment(Enum):
    Fibrosis, LymphNode, Calcium, Calcified, Bronchiectasis, Vessel = 0, 1, 2, 3, 4, 5


class Cancer500Nodule(NamedTuple):
    center_voxel: Sequence[int]
    review: Review
    comment: Optional[Comment] = None
    diameter_mm: Optional[float] = None
    texture: Optional[Texture] = None
    malignancy: Optional[bool] = None
