from hyperspy import api as hs
import numpy as np

from data4d import TVIPS, FourDimensionalData


def test_signal(signal4d):
    assert isinstance(signal4d, hs.signals.Signal2D)
    assert signal4d.data.ndim == 4
    assert signal4d.data.shape[:2] == (2, 2)
    assert signal4d.data.dtype == np.int16
    assert signal4d.metadata.Acquisition_instrument.TEM.beam_current == 21
    assert signal4d.metadata.Acquisition_instrument.TEM.beam_energy == 200
    assert signal4d.metadata.Acquisition_instrument.TEM.camera_length == 100


def test_tvips(tvips):
    assert tvips.stem.endswith("_000")
    assert tvips.suffix == ".tvips"
    t = TVIPS(tvips)
    assert isinstance(t, FourDimensionalData)
    assert t.scan_shape == (2, 2)
