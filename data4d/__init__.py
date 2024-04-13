from pathlib import Path

from .TVIPS import TVIPS
from .base import FourDimensionalData
from .blockfile import BLO

__version__ = open(Path(__file__).parent.joinpath("VERSION")).read().strip()
__all__ = ["FourDimensionalData", "BLO", "TVIPS"]
