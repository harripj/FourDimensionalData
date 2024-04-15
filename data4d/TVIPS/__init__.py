from ..utils import logger
from .tvipsfile import TVIPS

try:
    from .GUI.main import main as load_gui
except ImportError:
    logger.warning("GUI module not loaded")
