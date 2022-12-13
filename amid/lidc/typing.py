from enum import Enum
from typing import NamedTuple, Optional, Sequence

import numpy as np


class Calcification(Enum):
    Popcorn, Laminated, Solid, NonCentral, Central, Absent = 1, 2, 3, 4, 5, 6


class InternalStructure(Enum):
    SoftTissue, Fluid, Fat, Air = 1, 2, 3, 4


class Lobulation(Enum):
    NoLobulation, NearlyNoLobulation, MediumLobulation, NearMarkedLobulation, MarkedLobulation = 1, 2, 3, 4, 5


class Malignancy(Enum):
    HighlyUnlikely, ModeratelyUnlikely, Indeterminate, ModeratelySuspicious, HighlySuspicious = 1, 2, 3, 4, 5


class Sphericity(Enum):
    Linear, OvoidLinear, Ovoid, OvoidRound, Round = 1, 2, 3, 4, 5


class Spiculation(Enum):
    NoSpiculation, NearlyNoSpiculation, MediumSpiculation, NearMarkedSpiculation, MarkedSpiculation = 1, 2, 3, 4, 5


class Subtlety(Enum):
    ExtremelySubtle, ModeratelySubtle, FairlySubtle, ModeratelyObvious, Obvious = 1, 2, 3, 4, 5


class Texture(Enum):
    NonSolidGGO, NonSolidMixed, PartSolidMixed, SolidMixed, Solid = 1, 2, 3, 4, 5


class LIDCNodule(NamedTuple):
    center_voxel: Sequence[float]
    bbox: np.ndarray
    diameter_mm: float
    surface_area_mm2: float
    volume_mm3: float
    calcification: Optional[Calcification] = None
    internal_structure: Optional[InternalStructure] = None
    lobulation: Optional[Lobulation] = None
    malignancy: Optional[Malignancy] = None
    sphericity: Optional[Sphericity] = None
    spiculation: Optional[Spiculation] = None
    subtlety: Optional[Subtlety] = None
    texture: Optional[Texture] = None
