# this code is based on the Hyperspy implementation
import math
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

from PySide2.QtCore import Signal
from hyperspy.io_plugins.blockfile import get_header_dtype_list
from hyperspy.misc.array_tools import sarray2dict
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import constants
from tqdm.auto import tqdm

from ..base import FourDimensionalData
from ..utils import electron_wavelength

ENDIANESS = "<"
FRAME_OFFSET = 6  # each frame has 6 byte offset
IMAGE_CALIBRATION = 110  # pixels per cm on detector in standard ASTAR


def read_header(fname: Union[str, Path], endianess: str = ENDIANESS):
    """Read blockfile header and return as dict."""
    with open(fname, "rb") as f:
        header = sarray2dict(
            np.fromfile(f, dtype=get_header_dtype_list(endianess), count=1)
        )

    return header


def electron_voltage(header: dict) -> float:
    """Return electron beam voltage in kV."""
    return header["Beam_energy"] / 1_000


def pixel_size_scan(header: dict) -> Tuple[int, int]:
    """Return scan pixel size (i, j) (nm)."""
    return (header["SY"], header["SX"])


def pixel_size(
    header: dict, calibration: Union[int, float] = IMAGE_CALIBRATION
) -> float:
    f"""
    Return detector pixel size in 1/Angstrom. The default ASTAR
    configuration image calibration of {IMAGE_CALIBRATION} pixels per cm
    is used by default.
    """
    shape = frame_shape(header)
    if not len(set(shape)) == 1:
        print(f"Frame shape is not square: {shape}, using first axis for calculation.")

    cl = camera_length(header)
    # wavelength in Angstrom
    wavelength = electron_wavelength(electron_voltage(header))

    angular_radius = np.arctan(((shape[0] / 2.0) / calibration) / cl)
    # radius in 1/Angstrom
    # theta = lambda * r*
    radius = angular_radius / wavelength

    # pixel size is then radius in 1/Angstrom / no. pixels
    return radius / (shape[0] / 2.0)


def scan_shape(header: dict) -> Tuple[int, int]:
    """Return scan shape (i, j)."""
    # need to convert np ints to 64-bit to stop overflow
    return (int(header["NY"]), int(header["NX"]))


def number_of_frames(header: dict) -> int:
    """Return total number of frames within file."""
    return math.prod(scan_shape(header))


def frame_shape(header: dict) -> Tuple[int, int]:
    """Return frame shape (i, j). NB. assumed to be square."""
    # need to convert np ints to 64-bit to stop overflow
    return (int(header["DP_SZ"]), int(header["DP_SZ"]))


def camera_length(header: dict) -> float:
    """Return camera length in cm."""
    return header["Camera_length"] / 100


def frame_size(header: dict) -> int:
    """Return frame size in number of pixels."""
    return math.prod(frame_shape(header))


def frame_number_from_indices(header: dict, ij: NDArray) -> NDArray:
    """Get frame number from ij indices. ij has shape (2, N)."""
    return np.ravel_multi_index(ij, scan_shape(header))


def read_vbf(fname: Union[str, Path], endianess: str = ENDIANESS) -> NDArray:
    """Read the VBF reconstruction stored in the .blo file."""

    with open(fname, "rb") as f:
        header = sarray2dict(
            np.fromfile(f, dtype=get_header_dtype_list(endianess), count=1)
        )
        f.seek(header["Data_offset_1"])  # VBF information after this point

        out = np.fromfile(
            f,
            dtype=endianess + "u1",
            count=number_of_frames(header),
        ).reshape(scan_shape(header))

    return out


def _read_frame(
    fname: Union[str, Path],
    n: ArrayLike,
    header: Optional[dict] = None,
    endianess: str = ENDIANESS,
    progressbar: bool = True,
    return_index: bool = False,
    signal: Optional[Signal] = None,
) -> Generator[Union[NDArray, Tuple[NDArray, int]], None, None]:
    f"""
    Worker function to yield frames from .blo file.

    Parameters
    ----------
    fname: str or Path
        Path to file.
    n: array-like
        Frame numbers to read.
    header: None or dict
        .blo header dict if provided.
        Read from file if None.
    endianess: str
        Data ordering, typically {ENDIANESS}.
    progressbar: bool
        If True then a progressbar is shown.
    return_index: bool
        If True then each frame's index within the file is also returned.
    signal: Signal, optional
        If provided then a signal is emitted on each iteration.

    Yields
    ------
    frame or (frame, index) depending on the value of return_index.

    """
    if signal is not None and not hasattr(signal, "emit"):
        raise TypeError("signal must have .emit function, ie. QSignal.")

    with open(fname, "rb") as f:
        if header is None:
            header = sarray2dict(
                np.fromfile(f, dtype=get_header_dtype_list(endianess), count=1)
            )
        else:
            assert isinstance(header, dict), "header is not dict."

        fsize = frame_size(header)

        f.seek(header["Data_offset_2"])

        offset_previous = 0
        total = len(n)
        for i, num in enumerate(tqdm(n, disable=not progressbar)):
            offset = num * (fsize + FRAME_OFFSET) + FRAME_OFFSET
            frame = np.fromfile(
                f, dtype=endianess + "u1", count=fsize, offset=offset - offset_previous
            ).reshape(frame_shape(header))
            if return_index:
                out = (frame, num)
            else:
                out = frame
            yield out
            offset_previous = fsize + offset

            if signal is not None:
                signal.emit(int(100 * (i + 1) / total))


def read_frame(
    fname: Union[str, Path],
    n: ArrayLike,
    endianess: str = ENDIANESS,
    verbose: bool = True,
    return_index: bool = False,
    signal: Optional[Signal] = None,
) -> Union[NDArray, Generator[Union[NDArray, Tuple[NDArray, int]], None, None]]:
    """
    Read frame(s) from .blo file.

    Parameters
    ----------
    fname: str or Path
        Path to .blo file.
    n: int, array-like or None
        The sorted frame numbers to read.
        If None then all frames are read.
    endianess: str
        Endianess of the data.
    verbose: bool
        Display verbose information.
    return_index: bool
        If True then each frame's index within the file is also returned.
    signal: QSignal, optional
        If provided then a signal is emitted on each iteration.

    Returns or Yields
    -----------------
    frames: ndarray or generator
        A single frame is returned if n is a single number.
        Otherwise a generator of the frames is returned.

    """
    if n is None:
        n = range(number_of_frames(read_header(fname)))

    # make sure n is iterable
    if not hasattr(n, "__iter__"):
        n = (n,)
    # sort n to save on read time
    n = sorted(n)

    if len(n) == 1:
        progressbar = False
    else:
        progressbar = True if verbose else False

    # returns generator
    out = _read_frame(
        fname,
        n,
        endianess=endianess,
        progressbar=progressbar,
        return_index=return_index,
        signal=signal,
    )

    if len(n) == 1:
        # just get frame
        out = next(out)

    return out


class BLO(FourDimensionalData):
    def __init__(self, fname: Union[str, Path]):
        """Read data from ASTAR .blo blockfile."""
        self.header = read_header(fname)

        scan_shape = (int(self.header["NY"]), int(self.header["NX"]))
        frame_shape = (int(self.header["DP_SZ"]), int(self.header["DP_SZ"]))
        pixel_size_scan = (self.header["SY"], self.header["SX"])
        pixel_size_frame = (pixel_size(self.header, self.image_calibration),) * 2
        parameters = self.header
        dtype = next(_read_frame(fname, (0,), self.header, progressbar=False)).dtype

        super().__init__(
            fname,
            scan_shape,
            frame_shape,
            pixel_size_scan,
            pixel_size_frame,
            parameters,
            dtype,
        )

        self.vbf_intensities = self.vbf

    @property
    def electron_voltage(self) -> float:
        """Return electron beam voltage in kV."""
        return self.header["Beam_energy"] / constants.kilo

    @property
    def pixel_size(self) -> float:
        """Pixel size of diffraction pattern as in 1/Angstrom."""
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, x: float):
        self._pixel_size = x

    @property
    def number_of_frames(self) -> int:
        """Total number of frames in file."""
        return math.prod(self.scan_shape)

    @property
    def camera_length(self) -> float:
        """Return camera length in cm."""
        return self.header["Camera_length"] * constants.centi

    @property
    def image_calibration(self) -> float:
        """Return image (frame) calibration in pixels per cm."""
        return self.header["SDP"] * constants.centi

    @property
    def vbf(self) -> NDArray:
        """Return VBF image saved in blockfile."""
        with open(self.file, "rb") as f:
            f.seek(self.header["Data_offset_1"])  # VBF information after this point

            out = np.fromfile(
                f,
                dtype=ENDIANESS + "u1",
                count=self.number_of_frames,
            ).reshape(self.scan_shape)

        return out

    def read_frame(
        self,
        n: ArrayLike,
        endianess: str = ENDIANESS,
        verbose: bool = True,
        return_index: bool = False,
        signal: Optional[Signal] = None,
    ) -> Union[NDArray, Generator[Union[NDArray, Tuple[NDArray, int]], None, None]]:
        """
        Read frame(s) from .blo file.

        Parameters
        ----------
        fname: str or Path
            Path to .blo file.
        n: int, array-like or None
            The sorted frame numbers to read.
            If None then all frames are read.
        endianess: str
            Endianess of the data.
        verbose: bool
            Display verbose information.
        return_index: bool
            If True then each frame's index within the file is also returned.
        signal: QSignal, optional
            If provided then a signal is emitted on each iteration.

        Returns or Yields
        -----------------
        frames: ndarray or generator
            A single frame is returned if n is a single number.
            Otherwise a generator of the frames is returned.
        index: int
            If return_index is True then frame index within the file is also returned.
            Index is returned as tuple along with frame.

        """
        if n is None:
            n = range(self.number_of_frames)

        # make sure n is iterable
        if not hasattr(n, "__iter__"):
            n = (n,)
        # sort n to save on read time
        n = sorted(n)

        if len(n) == 1:
            progressbar = False
        else:
            progressbar = True if verbose else False

        # returns generator
        out = _read_frame(
            self.file,
            n,
            header=self.header,
            endianess=endianess,
            progressbar=progressbar,
            return_index=return_index,
            signal=signal,
        )

        return next(out) if len(n) == 1 else out
