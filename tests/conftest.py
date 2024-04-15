from pathlib import Path

from hyperspy import api as hs
from hyperspy.misc.utils import DictionaryTreeBrowser
import numpy as np
import pytest

from data4d import TVIPS


@pytest.fixture()
def data_dir() -> Path:
    path = Path(__file__).parent / "data"
    if not path.exists():
        path.mkdir()
    return path


@pytest.fixture()
def diffraction_metadata():
    metadata = {
        "Acquisition_instrument": {
            "TEM": {
                "beam_current": 21,
                "beam_energy": 200,
                "camera_length": 100,
            }
        },
        "General": {
            "date": "2024-04-01",
            "time": "12:03:05",
            "time_zone": "GMT",
        },
    }
    return DictionaryTreeBrowser(metadata)


@pytest.fixture
def signal4d(diffraction_metadata) -> hs.signals.Signal2D:
    im = hs.data.atomic_resolution_image()
    data = np.ascontiguousarray((np.tile(im, (2, 2, 1, 1)) * 2**16).astype(np.int16))
    signal = hs.signals.Signal2D(data)
    signal.metadata.add_dictionary(diffraction_metadata.as_dictionary())
    signal.axes_manager[0].scale_as_quantity = "1 nm"
    signal.axes_manager[1].scale_as_quantity = "1 nm"
    signal.axes_manager[2].scale_as_quantity = "1 1/nm"
    signal.axes_manager[3].scale_as_quantity = "1 1/nm"
    signal.save("/Users/paddyharrison/Downloads/test1.tvips")
    return signal


@pytest.fixture
def tvips(signal4d, data_dir) -> Path:
    fname = "test1_000.tvips"
    path = data_dir.joinpath(fname)
    if not path.exists():
        signal4d.save(path)
        t = TVIPS(path)
        t.scan_x = 2
        t.scan_y = 2
    return path
