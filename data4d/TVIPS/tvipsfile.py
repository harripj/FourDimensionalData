from datetime import datetime
import itertools
import logging
import math
import os
from pathlib import Path
from typing import IO, Any, Callable, Generator, Optional, Tuple, Union
import warnings

from PySide2.QtCore import Signal
import h5py
from hyperspy.io_plugins import blockfile
from ipywidgets import (
    Button,
    Checkbox,
    FloatRangeSlider,
    IntSlider,
    Output,
    VBox,
    interactive,
)
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
import tifffile
from tqdm.auto import tqdm

from ..base import FourDimensionalData
from ..utils import bin_box, find_direct_beam


class TVIPS(FourDimensionalData):
    """
    Class to handle .tvips data files from TVIPS camera systems.

    Based on code by Neils Cautaerts: https://github.com/din14970/TVIPSconverter
    """

    suffix = "tvips"
    # from TVIPS converter
    TVIPS_RECORDER_GENERAL_HEADER = [
        ("size", "u4"),  # unused - likely the size of generalheader in bytes
        ("version", "u4"),  # 1 or 2
        ("dimx", "u4"),  # dp image size width
        ("dimy", "u4"),  # dp image size height
        ("bitsperpixel", "u4"),  # 8 or 16
        ("offsetx", "u4"),  # generally 0
        ("offsety", "u4"),
        ("binx", "u4"),  # camera binning
        ("biny", "u4"),
        ("pixelsize", "u4"),  # nm, physical pixel size
        ("ht", "u4"),  # high tension, voltage
        ("magtotal", "u4"),  # magnification/camera length?
        ("frameheaderbytes", "u4"),  # number of bytes per frame header
        ("dummy", "S204"),  # just writes out TVIPS TVIPS TVIPS
    ]

    TVIPS_RECORDER_FRAME_HEADER = [
        ("num", "u4"),  # seems to cycle also
        ("timestamp", "u4"),  # seconds since 1.1.1970
        ("ms", "u4"),  # additional milliseconds to the timestamp
        ("LUTidx", "u4"),  # always the same value
        ("fcurrent", "f4"),  # 0 for all frames
        ("mag", "u4"),  # same for all frames
        ("mode", "u4"),  # 1 -> image 2 -> diff
        ("stagex", "f4"),
        ("stagey", "f4"),
        ("stagez", "f4"),
        ("stagea", "f4"),
        ("stageb", "f4"),
        ("rotidx", "u4"),
        ("temperature", "f4"),  # cycles between 0.0 and 9.0 with step 1.0
        ("objective", "f4"),  # kind of randomly between 0.0 and 1.0
        # for header version 2, some more data might be present
    ]

    _max_frames = 1_000_000  # maximum number of frames, in reality it is much less

    def __init__(self, fname: Union[str, Path]):
        """
        Open a .tvips file. This file may be in more than one part, in which case provide the first file.

        Parameters
        ----------
        fname: str or Path
            Path to first .tvips file in series.

        """
        # check for extra files
        self.files = self._check_for_files(fname)
        # get file sizes in bytes
        self._file_sizes = [os.path.getsize(f) for f in self.files]
        self._cumulative_file_sizes = [
            sum(self._file_sizes[:i]) for i in range(len(self._file_sizes) + 1)
        ]

        self._read_general_header(self.files[0])
        # guess number of frames in both files
        self._calculate_number_of_frames()

        # initialise vbf and direct beam coordinates
        self._fname_tvipsinfo = self.files[0].parent.joinpath(
            os.extsep.join((self._series, "tvipsinfo"))
        )
        if not self._fname_tvipsinfo.exists():
            with h5py.File(self._fname_tvipsinfo, "w") as h5:
                h5.create_group("Scan Parameters")
            print(f".recon file created: {self._fname_tvipsinfo}.")

        scan_parameters = self.scan_parameters
        try:
            scan_shape = (scan_parameters["ni"], scan_parameters["nj"])
        except KeyError:
            scan_shape = (self.number_of_frames, 1)

        frame_shape = (self.general_header["dimy"], self.general_header["dimx"])
        pixel_size_scan = (
            self.general_header["pixelsize"],
            self.general_header["pixelsize"],
        )
        pixel_size_frame = (1, 1)
        parameters = self.general_header
        if self.general_header["bitsperpixel"] == 8:
            dtype = np.uint8
        elif self.general_header["bitsperpixel"] == 16:
            dtype = np.uint16
        else:
            dtype = None

        super().__init__(
            self.files[0],
            scan_shape,
            frame_shape,
            pixel_size_scan,
            pixel_size_frame,
            parameters,
            dtype,
        )

    @staticmethod
    def _check_input_file(fname: Union[str, Path]):
        fname = Path(fname)

        if not fname.exists():
            raise IOError(f"File does not exist: {fname}.")

        if fname.suffix.lower() != os.extsep + TVIPS.suffix:
            raise ValueError(
                f"File is not a tvips file, extension incorrect: f{fname}."
            )

        return fname

    @property
    def vbf_intensities(self) -> NDArray:
        """Get VBF intensities from .tvipsinfo file.
        Calculates VBF if not found."""
        try:
            with h5py.File(self._fname_tvipsinfo, "r") as h5:
                out = h5["VBF Intensities"][:]
        except KeyError:
            # calculate vbf
            self.calculate_virtual_bright_field_reconstruction()
            # get data
            out = self._vbf_intensities
            # set in .hdf
            self.vbf_intensities = out

        return out[
            self.scan_offset : self.scan_offset + np.prod(self.scan_shape)
        ].reshape(self.scan_shape)

    @vbf_intensities.setter
    def vbf_intensities(self, x: ArrayLike):
        x = np.asarray(x)
        if x.size != self.number_of_frames:
            raise ValueError(
                f"x is not defined for each frame: {x.size} != {self.number_of_frames}."
            )
        with h5py.File(self._fname_tvipsinfo, "r+") as h5:
            if "VBF Intensities" in h5.keys():
                h5["VBF Intensities"][...] = x
            else:
                h5["VBF Intensities"] = x

    @property
    def direct_beam_coordinates(self) -> NDArray:
        with h5py.File(self._fname_tvipsinfo, "r+") as h5:
            try:
                out = h5["Direct Beam Coordinates"][:]
                print(f"Direct Beam Coordinates read from {self._fname_tvipsinfo}.")
            except KeyError:
                out = np.full(
                    (self.number_of_frames, len(self.frame_shape)), -1, dtype=int
                )
                self.direct_beam_coordinates = out
        return out

    @direct_beam_coordinates.setter
    def direct_beam_coordinates(self, data: NDArray):
        with h5py.File(self._fname_tvipsinfo, "r+") as h5:
            if "Direct Beam Coordinates" in h5.keys():
                h5["Direct Beam Coordinates"][...] = data
            else:
                h5["Direct Beam Coordinates"] = data

    @property
    def scan_parameters(self) -> dict:
        with h5py.File(self._fname_tvipsinfo, "r") as h5:
            try:
                out = dict(**h5["Scan Parameters"].attrs)
            except KeyError:
                out = None
        return out

    @scan_parameters.setter
    def scan_parameters(self, parameters: dict):
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be dict")
        with h5py.File(self._fname_tvipsinfo, "r+") as h5:
            try:
                grp = h5["Scan Parameters"]
            except KeyError:
                grp = h5.create_group("Scan Parameters")
            for k, v in parameters.items():
                grp.attrs[k] = v

    @property
    def frame_total(self) -> NDArray:
        try:
            with h5py.File(self._fname_tvipsinfo, "r") as h5:
                out = h5["Frame total"][:]
        except KeyError:
            out = super().frame_total
            self.frame_total = out  # add to tvipsinfo .hdf file

        return out[
            self.scan_offset : self.scan_offset + np.prod(self.scan_shape)
        ].reshape(self.scan_shape)

    @frame_total.setter
    def frame_total(self, x: ArrayLike):
        x = np.asarray(x)
        if x.size != self.number_of_frames:
            raise ValueError(
                f"x is not defined for each frame: {x.size} != {self.number_of_frames}."
            )
        key = "Frame total"
        with h5py.File(self._fname_tvipsinfo, "r+") as h5:
            if key in h5:
                h5[key][...] = x
            else:
                h5[key] = x

    @property
    def scan_offset(self) -> int:
        try:
            offset = self.scan_parameters["offset"]
        except KeyError:
            offset = 0
        return offset

    @scan_offset.setter
    def scan_offset(self, i: int):
        if not isinstance(i, int):
            raise TypeError("offset must be an integer")
        self.scan_parameters = dict(offset=i)

    @FourDimensionalData.scan_x.setter
    def scan_x(self, x: int):
        if not isinstance(x, int):
            raise TypeError("x must be an integer")
        self.scan_shape = (self.scan_y, x)
        self.scan_parameters = dict(nj=x)

    @FourDimensionalData.scan_y.setter
    def scan_y(self, y: int):
        if not isinstance(y, int):
            raise TypeError("y must be an integer")
        self.scan_shape = (y, self.scan_x)
        self.scan_parameters = dict(ni=y)

    def _check_for_files(self, fname: Union[str, Path]) -> list:
        fname = self._check_input_file(fname)

        # if overflow_files is not None:
        #     if isinstance(overflow_files, str, Path):
        #         out.append(overflow_files)
        #     elif isinstance(overflow_files, (list, tuple)):
        #         for o in overflow_files:
        #             out.append(o)
        #     else:
        #         raise ValueError(
        #             "overflow_files should be either str, Path, or list or tuple of paths."
        #         )

        files = fname.parent.glob("*" + str(fname.suffix))
        # match the experimental base of each found file and add to list
        self._series = "_".join(str(fname.stem).split("_")[:-1])
        out = [f for f in files if self._series in f.stem]

        # now sort by name
        out = list(map(self._check_input_file, sorted(out)))

        # check that first file has index 000
        if int(out[0].stem.split("_")[-1]):
            logging.info(
                f"File does not appear to be initial experimental file (typically ends in 000: {fname}."
            )

        return out

    @property
    def number_of_frames(self):
        return self._calculate_number_of_frames()

    def _calculate_number_of_frames(self) -> int:
        """Estimate number of frames in all files knowing the frame and header sizes."""
        n = math.ceil(
            (
                self._cumulative_file_sizes[-1]
                - len(self.files) * self._general_header_size
            )
            / (self._frame_size_bytes + self.general_header["frameheaderbytes"])
        )
        return n

    def _read_general_header(self, file: Union[str, Path]):
        with tifffile.FileHandle(file, "rb") as f:
            keys = (i[0] for i in self.TVIPS_RECORDER_GENERAL_HEADER)
            dtypes = (i[1] for i in self.TVIPS_RECORDER_GENERAL_HEADER)
            general = dict(zip(keys, f.read_record(self.TVIPS_RECORDER_GENERAL_HEADER)))

            self._general_header_size = sum(np.dtype(i).itemsize for i in dtypes)
            self.frame_shape = (general["dimy"], general["dimx"])
            self._frame_size_bytes = (
                math.prod(self.frame_shape) * general["bitsperpixel"] // 8
            )
            try:
                self.dtype = getattr(np, f"uint{general['bitsperpixel']}")
            except Exception as err:
                print(
                    f"dtype with bitsperpixel: {general['bitsperpixel']} not understood."
                )
                raise err
            self.general_header = general

    def _read_frame_header(self, fh: IO):
        keys = (i[0] for i in self.TVIPS_RECORDER_FRAME_HEADER)
        dtypes = (i[1] for i in self.TVIPS_RECORDER_FRAME_HEADER)
        self._frame_header_size = sum(np.dtype(i).itemsize for i in dtypes)

        return dict(
            zip(
                keys,
                fh.read_record(self.TVIPS_RECORDER_FRAME_HEADER),
            )
        )

    def _format_frame_numbers(self, n: Optional[ArrayLike]):
        """
        Checks frame numbers n to see if they are of the right format.

        Parameters
        ----------
        n: int, iterable, or None
            The frame number(s) within the file.
            If None then all files are read.

        Returns
        -------
        n: iterable

        """

        if n is None:
            try:
                _max = self.number_of_frames
            except AttributeError:
                _max = self._max_frames
            n = range(_max)
        elif isinstance(n, (int, np.integer)):
            n = (n,)
        elif hasattr(n, "__iter__"):
            pass
        else:
            raise ValueError(f"n {type(n)} should be either int, iterable, or None.")

        return n

    def _get_frame_header_start_byte(self, n: ArrayLike) -> Generator[int, None, None]:
        """
        Return frame header start byte position within file.

        Parameters
        ----------
        n: int, array-like, or None
            The frame number(s) within the file.
            If None then all files are read.

        Yields
        ------
        i: integer
            Frame header start byte for each frame.

        """
        n = self._format_frame_numbers(n)

        for i in n:
            yield self._general_header_size + i * (
                self.general_header["frameheaderbytes"] + self._frame_size_bytes
            )

    def _get_frame_start_byte(self, n: ArrayLike) -> Generator[int, None, None]:
        """
        Return frame start byte position within file.

        Parameters
        ----------
        n: int, array-like, or None
            The frame number(s) within the file.
            If None then all files are read.

        Yields
        ------
        i: integer
            Frame start byte for each frame.

        """
        n = self._format_frame_numbers(n)

        for i in self._get_frame_header_start_byte(n):
            yield i + self.general_header["frameheaderbytes"]

    def _data_in_file_index(self, byte: int) -> Optional[int]:
        """
        Check which file the data requested is in.

        Parameters
        ----------
        byte: int
            The start byte position of the data (assuming one linear file).

        Returns
        -------
        index: int
            The file index within self.files.

        """
        # happens when reading all files
        if byte > self._cumulative_file_sizes[-1]:
            return None
        else:
            return tuple(byte < size for size in self._cumulative_file_sizes[1:]).index(
                True
            )

    def _read_frame(
        self,
        n: ArrayLike,
        return_header: bool = False,
        return_index: bool = False,
        memmap: bool = False,
        signal: Optional[Signal] = None,
    ) -> Generator[
        Union[NDArray, Tuple[NDArray, dict], Tuple[NDArray, dict, int]], None, None
    ]:
        """
        Read one or more frames from file. This private function does the heavy lifting.
        The wrapper with the same name allows for a return call if n is just one number.

        Parameters
        ----------
        n: int or interable of ints, or None
            The frames to return, n should be sorted.
            If None then all frames will be read.
        return_header: bool
            If True then each frame's header is also returned.
        return_index: bool
            If True then each frame's index within the file is also returned.
        signal: None or QSignal
            If provided the signal will emit on each iteration.

        Yields
        ------
        frame or (frame, header, index) depending on the value of return_header and return_index.

        """
        if signal is not None and not hasattr(signal, "emit"):
            raise TypeError("signal must have .emit function, ie. QSignal.")

        n = tuple(self._format_frame_numbers(n))
        total = len(n)

        startbytes = tuple(
            b
            for b in self._get_frame_start_byte(n)
            if b < self._cumulative_file_sizes[-1]
        )
        # find out which frame is in which file
        file_index = tuple(self._data_in_file_index(b) for b in startbytes)

        # want to open each file only once...
        for fi in set(file_index):
            # filter startbytes
            selector = map(lambda x: x == fi, file_index)
            with tifffile.FileHandle(self.files[fi], "rb") as f:
                # get number of bytes in previous files before this one
                file_start = self._cumulative_file_sizes[fi]
                # filter only frames within this file and get startbytes
                # remove bytes from previous files from these byte values
                startbytes_file = (
                    sb - file_start for sb in itertools.compress(startbytes, selector)
                )
                # now loop over and get frames...
                for i, (s, num) in enumerate(
                    tqdm(
                        zip(startbytes_file, n),
                        disable=True if len(n) < 2 else False,
                        desc=self.files[fi].stem,
                        total=total,
                    )
                ):
                    offset = s - f.tell()
                    if return_header:
                        f.seek(offset - self.general_header["frameheaderbytes"], 1)
                        header = self._read_frame_header(f)
                        offset = (
                            self.general_header["frameheaderbytes"]
                            - self._frame_header_size
                        )
                    # read frame
                    if memmap:
                        frame = np.memmap(
                            f,
                            dtype=self.dtype,
                            mode="r",
                            shape=self.frame_shape,
                            offset=s,
                        )
                    else:
                        frame = np.fromfile(
                            f,
                            dtype=self.dtype,
                            count=math.prod(self.frame_shape),
                            offset=offset,
                        ).reshape(self.frame_shape)

                    # return frame generator with optional header and index
                    if return_header or return_index:
                        out = [frame]
                        if return_header:
                            out.append(header)
                        if return_index:
                            out.append(num)
                    else:
                        out = frame

                    if signal is not None:
                        signal.emit(int(100 * (i + 1) / total))

                    yield out

    def read_frame(
        self,
        n: ArrayLike,
        return_header: bool = False,
        return_index: bool = False,
        memmap: bool = False,
        signal: Optional[Signal] = None,
    ) -> Union[
        Union[NDArray, Tuple[NDArray, dict], Tuple[NDArray, dict, int]],
        Generator[
            Union[NDArray, Tuple[NDArray, dict], Tuple[NDArray, dict, int]], None, None
        ],
    ]:
        """
        Read one or more frames from file.

        Parameters
        ----------
        n: int or interable of ints, or None
            The frames to return, n should be sorted.
            If None then all frames will be read.
        return_header: bool
            If True then each frame's header is also returned.
        return_index: bool
            If True then each frame's index within the file is also returned.
        signal: None or QSignal
            If provided the signal will emit on each iteration.

        Yields or Returns
        -----------------
        frame or (frame, header) depending on frame_header.
        Returns if n is just one numeber, otherwise yields a generator.

        """

        n = tuple(self._format_frame_numbers(n))

        frames = self._read_frame(
            n,
            return_header=return_header,
            return_index=return_index,
            memmap=memmap,
            signal=signal,
        )

        _len_n = len(n)
        # should not be 0
        assert _len_n, f"n with length {_len_n} should not be possible..."

        if _len_n == 1:
            return next(frames)
        else:
            return frames

    @staticmethod
    def recenter_mask(
        mask: ArrayLike, new_center: Tuple[int, int], old_center: Tuple[int, int]
    ) -> Union[NDArray, Tuple[NDArray]]:
        """
        Shift a mask to a new center position.

        Parameters
        ----------
        mask: (M, N) ndarray or (N, ndim) tuple of arrays
            If mask is ndarray, it must be the same shape as frame.
            If tuple or arrays, these arrays represent coordinates within the frame.
            Note that if mask is ndarray the mask is just rolled through shift.
        new_center, old_center: iterable of ints
            The new and old center location of the mask.
            The shift is calculated as the difference between these two values.
            Must have the same length as mask.ndim.

        Returns
        -------
        mask: (M, N) ndarray or (N, ndim) tuple of ints
            The new mask.

        """
        shift = tuple(new - old for new, old in zip(new_center, old_center))

        # check inputs
        if isinstance(mask, np.ndarray):
            out = np.roll(mask, shift, axis=range(len(new_center)))
        elif isinstance(mask, tuple):
            out = tuple(
                (mask[i] + shift[i]).round().astype(int) for i in range(len(new_center))
            )
        else:
            raise ValueError("mask must be ndarray or tuple of integer arrays.")

        return out

    def recenter_frame(
        self, frame: NDArray, center: Tuple[int, int], mode: str = "edge"
    ) -> NDArray:
        """
        Recenter frame by rolling. If center is comprised of floats then subpixel rolling will be performed.

        Parameters
        ----------
        frame: ndarray
            Frame to roll.
        center: (frame.ndim,) tuple
            The center coordinates of the frame.
        mode: str
            Passed to np.roll and np.pad.

        Returns
        -------
        out: ndarray
            The centered frame.

        """

        shift = (f // 2 - c for f, c in zip(frame.shape, center))

        if all(isinstance(c, (int, np.integer)) for c in center):
            return np.roll(frame, shift, axis=range(len(shift)), mode=mode)
        else:
            return self.roll_array_subpixel(frame, shift, mode=mode)

    def calculate_quantity_over_patterns(
        self,
        n: Optional[ArrayLike],
        fn: Callable,
        recenter: bool = False,
        key: Optional[str] = None,
    ) -> NDArray:
        """
        Calculate some quantity over n frames.

        Each frame is loaded successively and stacked with the previous result.
        A fn is then applied: fn(stack, axis=0), so fn must accept keyword axis=0.

        Parameters
        ----------
        n: iter, int, or None
            The frames to use, use None for whole file.
        fn: func
            The function to apply to the stack of two images.
            fn must accept keyword axis=0, eg np.sum, np.max.
        recenter: bool
            If True the frame is recentered before being stacked.
        key: str or None
            If str then the final output is added to .recon as key.

        Returns
        -------
        out: ndarray
            The fn applied to all the desired frames.

        """
        if recenter:
            dbc = self.direct_beam_coordinates

        for i, (frame, n_) in enumerate(self.read_frame(n, return_index=True)):
            if recenter:
                if (dbc[n_] < 0).any():
                    center = find_direct_beam(frame)
                    dbc[n_] = center
                else:
                    center = dbc[n_]
                frame = self.recenter_frame(frame, center)

            if not i:
                temp = frame
            else:
                temp = fn(np.stack((frame, temp), axis=0), axis=0)

        if recenter:
            self.direct_beam_coordinates = dbc

        return temp

    # def calculate_virtual_reconstruction(
    #     self, mask, n=None, recenter=False, side=50, sigma=3.0
    # ):
    #     """

    #     Calculate virtual reconstruction from a mask.

    #     Parameters
    #     ----------
    #     mask: (M, N) ndarray or (N, ndim) tuple of arrays
    #         If mask is ndarray, it must be the same shape as frame.
    #         If tuple or arrays, these arrays represent coordinates within the frame.
    #     n: None or array-like
    #         Frame numbers to compute.
    #         If None then all frames are computed.
    #     recenter: bool
    #         If True then the direct beam position is computed and the mask is recentered on this location.
    #         It is assumed that mask is initially centered on the center of each frame.
    #     side: int
    #         Box side size for computing the central beam location.
    #     sigma: float
    #         Gaussian parameter for the recentering algorithm.

    #     Returns
    #     -------
    #     intensities: (n,) ndarray-like
    #         The reconstructed intensities.

    #     """
    #     # check inputs
    #     if isinstance(mask, np.ndarray):
    #         assert (
    #             mask.shape == self.frame_shape
    #         ), f"mask shape {mask.shape} does not equal frame shape {self.frame_shape}."
    #     elif isinstance(mask, tuple):
    #         assert len(mask) == len(
    #             self.frame_shape
    #         ), f"mask has incorrect dimensionality {len(mask)}."
    #     else:
    #         raise ValueError("mask must be ndarray or tuple of integer arrays.")

    #     # load once to avoid opening file many times
    #     dbc = self.direct_beam_coordinates
    #     # init output
    #     intensities = []

    #     for i, (frame, n_) in enumerate(self.read_frame(n, return_index=True)):
    #         if recenter:
    #             # if the data has not been written, ie. -1
    #             if (dbc[n_] < 0).any():
    #                 center = find_direct_beam(frame, side, sigma)
    #                 # update center_location and unmask array
    #                 dbc[n_] = center
    #             # otherwise read
    #             else:
    #                 center = dbc[n_]
    #             mask_temp = self.recenter_mask(
    #                 mask, center, tuple(i // 2 for i in frame.shape)
    #             )
    #         else:
    #             mask_temp = mask

    #         intensities.append(frame[mask_temp].sum())

    #     if recenter:
    #         # write to file after using all frames
    #         self.direct_beam_coordinates = dbc

    #     return intensities

    # def __len__(self):
    #     return self.number_of_frames

    # @staticmethod
    # def _calculate_frame_index(ij, shape, offset=0, crop=(0, 0)):
    #     """
    #     Calculate the sequential frame number within the tvips file produced by TVIPSconverter.

    #     Parameters
    #     ----------
    #     ij: tuple or (N, 2) integer array
    #         Tuple of length len(scan_shape), ie. one coordinate for each dimension.
    #         Otherwise array where second axis has length len(scan_shape), ie. columns of row (i) and column (j) (etc.) of indices in question.
    #     shape: tuple
    #         Overall shape of experimental scan (from TVIPS).
    #     crop: tuple
    #         The ij of the initial crop position if querying the location of a pixel within cropped data of the original scan.
    #     offset: int
    #         Initial scan frame offset within experimental data (from TVIPS).

    #     Returns
    #     -------
    #     indices: int
    #         Frame index within experimental .tvips file.

    #     """

    #     # if just one frame, use tuples etc... it is easier
    #     if isinstance(ij, (list, tuple)) and isinstance(ij[0], int):
    #         index = (ij[0] + crop[0]) * shape[1] + ij[1] + crop[1] + offset
    #         return index
    #     else:
    #         return (
    #             np.ravel_multi_index(np.transpose(np.asarray(ij) + crop), shape)
    #             + offset
    #         )

    # def calculate_frame_index(self, ij, shape=None, offset=None, crop=(0, 0)):
    #     """
    #     Calculate the sequential frame number within the tvips file produced by TVIPSconverter.

    #     Parameters
    #     ----------
    #     ij: tuple or (N, 2) integer array
    #         Tuple of length len(scan_shape), ie. one coordinate for each dimension.
    #         Otherwise array where second axis has length len(scan_shape), ie. columns of row (i) and column (j) (etc.) of indices in question.
    #     shape: tuple
    #         Overall shape of experimental scan (from TVIPS).
    #     crop: tuple
    #         The ij of the initial crop position if querying the location of a pixel within cropped data of the original scan.
    #     offset: int
    #         Initial scan frame offset within experimental data (from TVIPS).

    #     Returns
    #     -------
    #     indices: int
    #         Frame index within experimental .tvips file.

    #     """
    #     # use stored parameters by default
    #     if shape is None or offset is None:
    #         ni, nj, offset = self.shape_and_offset
    #     if shape is None:
    #         shape = (ni, nj)
    #     if offset is None:
    #         offset = offset

    #     return self._calculate_frame_index(ij, shape, offset, crop)

    @staticmethod
    def save_reconstruction(
        fname: Union[str, Path],
        key: str,
        arr: NDArray = None,
        **parameters: dict[str, Any],
    ):
        """
        Add reconstruction and parameters as attrs at key within .recon.
        """
        with h5py.File(fname, "r+") as h5:
            if arr is not None:
                if key in h5:
                    h5[key][...] = arr
                else:
                    h5[key] = arr
            for k, v in parameters.items():
                h5[key].attrs[k] = v
            print(f"Reconstruction {key} saved: {fname}.")

    # def calculate_virtual_bright_field_reconstruction(
    #     self,
    #     radius=10,
    #     n=None,
    #     recenter=False,
    #     side=50,
    #     sigma=3.0,
    #     load=True,
    #     save=True,
    # ):
    #     """

    #     Calculate virtual bright field reconstruction.

    #     Parameters
    #     ----------
    #     radius: float
    #         The VBF aperture radius in pixels.
    #     n: None or array-like
    #         Frame numbers to compute.
    #         If None then all frames are computed.
    #     recenter: bool
    #         If True then the direct beam position is computed and the mask is recentered on this location.
    #         It is assumed that mask is initially centered on the center of each frame.
    #     side: int
    #         Box side size for computing the central beam location.
    #     sigma: float
    #         Gaussian parameter for the recentering algorithm.
    #     io: bool
    #         If True then load vbf file is available and save file if calculated.
    #         The .vbf file will be read from the .tvips file parent's directory with the same series name and .vbf extension.

    #     Returns
    #     -------
    #     intensities: (n,) ndarray-like
    #         The reconstructed intensities.

    #     """

    #     if not (
    #         load and self._fname_recon.exists() and self.vbf_intensities is not None
    #     ):
    #         # compute vbf
    #         center = (i / 2.0 for i in self.frame_shape)
    #         coords = draw.disk(center, radius=radius, shape=self.frame_shape)

    #         self.vbf_intensities = self.calculate_virtual_reconstruction(
    #             coords, n, recenter=recenter, side=side, sigma=sigma
    #         )

    #         if save:
    #             self.save_reconstruction(
    #                 self._fname_recon,
    #                 "VBF Intensities",
    #                 self.vbf_intensities,
    #                 recenter=recenter,
    #                 radius=radius,
    #             )

    #     return self.vbf_intensities

    def estimate_shape(
        self,
        n: int = 2_500,
        comparator: Callable = np.less_equal,
        order: int = 100,
        ax: Optional[Axes] = None,
    ):
        """
        Estimate shape according to some periodicity in the VBF intensities.

        Parameters
        ----------
        n: int
            Number of VBF intensities to sample from start of file.
        comparator, order
            See scip.signal.argrelextrema.
        ax: None or plt.Axes
            If provided then the calculation will be plotted on an Axes.
            Helps to visualize.

        Returns
        -------
        shape: tuple
            The estimated scan shape.

        """
        peaks = signal.argrelextrema(self.vbf_intensities[:n], comparator, order=order)[
            0
        ]
        j = int(np.ediff1d(peaks).mean())
        i = int(len(self) / j)

        if isinstance(ax, Axes):
            ax.plot(self.vbf_intensities[:n])
            for p in peaks:
                ax.axvline(p, color="gray", ls="dashed")

        return (i, j)

    def plot(
        self,
        shape: Optional[Tuple] = None,
        figsize: Tuple[int, int] = (12, 6),
        shape_max: Tuple[int, int] = (512, 512),
    ):
        """
        Convenience plotting function to plot data interactively.
        """

        fig, ax = plt.subplots(ncols=2, figsize=figsize)

        if self.vbf_intensities is None:
            # default values...
            print("Calculating VBF reconstruction with default values...")
            self.calculate_virtual_bright_field_reconstruction()

        if self.scan_parameters:
            # used for initial slider values
            print(f"Using reconstruction parameters: {self.scan_parameters}.")
            try:
                shape = (
                    self.scan_parameters["ni"],
                    self.scan_parameters["nj"],
                )
                offset = self.scan_parameters["offset"]
            except KeyError:
                # use default values
                offset = 0
        else:
            print("Estimating shape and setting offset to 0...")
            shape = self.estimate_shape()
            offset = 0

        if math.prod(shape) > self.number_of_frames:
            raise ValueError(
                f"Not enough frames (total {self.number_of_frames}) for shape {shape}."
            )

        # initial images
        vbf_image = ax[0].matshow(
            np.reshape(self.vbf_intensities[: math.prod(shape)], shape), cmap="magma"
        )
        frame = self.read_frame(0)
        diffraction_image = ax[1].matshow(frame)

        crosshair_vertical = ax[0].axvline(0, color="k", alpha=0.4)
        crosshair_horizontal = ax[0].axhline(0, color="k", alpha=0.4)

        # define some useful widgets
        i = IntSlider(0, 0, 1, description="i")
        j = IntSlider(0, 0, 1, description="j")
        ni = IntSlider(shape[0], 1, shape_max[0], description="ni")
        nj = IntSlider(shape[1], 1, shape_max[1], description="nj")
        offset = IntSlider(offset, 0, self.number_of_frames, description="offset")
        log = Checkbox(True)
        clim = Checkbox(True, description="Auto clim?")

        # check frame type
        if np.issubdtype(frame.dtype, np.integer):
            info = np.iinfo(frame.dtype)
        elif np.issubdtype(frame.dtype, np.floating):
            info = np.finfo(frame.dtype)
        climrange = FloatRangeSlider(
            value=(0, 127),
            min=0,
            max=1023,  # info.max,
            description="Manual limits:",
            orientation="horizontal",
        )

        def update_vbf_parameters(button):
            parameters = dict(ni=ni.value, nj=nj.value, offset=offset.value)
            self.scan_parameters = parameters
            with output:
                print(f"VBF parameters written: {parameters}.")

        button_save = Button(description="Update .recon")
        button_save.on_click(update_vbf_parameters)
        output = Output()

        def update(
            ii: int,
            jj: int,
            nii: int,
            njj: int,
            offsett: int,
            log: bool,
            clim: bool,
            climrange: Tuple[float, float],
        ):
            # change slider limits to avoid over reading data
            i.max = nii
            j.max = njj

            shape = (nii, njj)
            offset.max = self.number_of_frames - math.prod(shape)

            # check shape
            if math.prod(shape) + offsett > self.number_of_frames:
                print(
                    f"Too many frames requested for total data size {self.number_of_frames}.",
                    end="\t\t\r",
                )
            else:
                crosshair_vertical.set_xdata((jj, jj))
                crosshair_horizontal.set_ydata((ii, ii))

                vbf = np.reshape(
                    self.vbf_intensities[offsett : offsett + math.prod(shape)], shape
                )
                vbf_image.set_array(vbf)

                index = offsett + ii * njj + jj
                diffraction = self.read_frame(index)
                if log:
                    diffraction = np.log(diffraction - diffraction.min() + 1)
                diffraction_image.set_array(diffraction)

                # set vbf image clim regardless
                vbf_image.set_clim(vbf.min(), vbf.max())
                if clim:
                    diffraction_image.set_clim(diffraction.min(), diffraction.max())
                else:
                    diffraction_image.set_clim(climrange[0], climrange[1])

        mouse_pressed = False
        mouse_in_axes = False

        # add mouse functions
        def mouse_clicked(event):
            nonlocal mouse_pressed
            mouse_pressed = True

        def mouse_released(event):
            nonlocal mouse_pressed
            mouse_pressed = False

        def axes_enter(event):
            nonlocal mouse_in_axes
            mouse_in_axes = True

        def axes_leave(event):

            nonlocal mouse_in_axes
            mouse_in_axes = False

        def mouse_dragged(event):
            nonlocal mouse_pressed, mouse_in_axes
            if mouse_in_axes and mouse_pressed:
                i.value = int(event.ydata)
                j.value = int(event.xdata)

        cid = []
        cid.append(fig.canvas.mpl_connect("button_press_event", mouse_clicked))
        cid.append(fig.canvas.mpl_connect("button_release_event", mouse_released))
        cid.append(fig.canvas.mpl_connect("figure_enter_event", axes_enter))
        cid.append(fig.canvas.mpl_connect("figure_leave_event", axes_leave))
        cid.append(fig.canvas.mpl_connect("motion_notify_event", mouse_dragged))

        return VBox(
            (
                interactive(
                    update,
                    ii=i,
                    jj=j,
                    nii=ni,
                    njj=nj,
                    offsett=offset,
                    log=log,
                    clim=clim,
                    climrange=climrange,
                ),
                button_save,
                output,
            )
        )

    def get_blo_header(
        self,
        binning: int = 1,
        scan_scale: float = 1.0,
        ppcm: Optional[float] = None,
        endianess: str = "<",
        **kwargs: dict[str, Any],
    ):
        header = blockfile.get_default_header(endianess)
        note = ""

        scan_shape = self.scan_shape
        if len(scan_shape) == 2:
            NY, NX = scan_shape  # flip ij -> xy
            SX = scan_scale
            SY = scan_scale
        elif len(scan_shape) == 1:
            NX = scan_shape[0]
            NY = 1
            SX = scan_scale
            SY = scan_scale
        else:
            raise ValueError("Invalid data shape")

        DP_SZ = tuple(s // binning for s in self.frame_shape)

        if DP_SZ[0] != DP_SZ[1]:
            raise ValueError("Blockfiles require DP shape to be square!")
        DP_SZ = DP_SZ[0]

        # use default values for TVIPS TemCam-XF416, ie. 63.5mm x 63.5mm FoV
        if ppcm is None:
            ppcm = DP_SZ / 6.35
        SDP = 100.0 * ppcm

        offset2 = NX * NY + header["Data_offset_1"]
        # Based on inspected files, the DPs are stored at 16-bit boundary...
        # Normally, you'd expect word alignment (32-bits) ¯\_(°_o)_/¯
        offset2 += offset2 % 16

        header_sofar = {
            "NX": NX,
            "NY": NY,
            "DP_SZ": DP_SZ,
            "SX": SX,
            "SY": SY,
            "SDP": SDP,
            "Data_offset_2": offset2,
        }

        header_sofar.update(kwargs)

        # read first frame and header to get acq. time
        frame, frame_header = self.read_frame(0, return_header=True)
        acq_time = datetime.fromtimestamp(frame_header["timestamp"])
        header_sofar["Acquisition_time"] = blockfile.datetime_to_serial_date(acq_time)

        header = blockfile.dict2sarray(header_sofar, sarray=header)

        return header, note

    def create_blo_file(
        self,
        filename: Optional[Union[str, Path]] = None,
        binning: int = 1,
        clip: Optional[Tuple[int, int]] = None,
        check_exists: bool = True,
        signal: Optional[Signal] = None,
        **kwds: dict[str, Any],
    ):
        """
        Create a .blo file from TVIPS data.

        Parameters
        ----------
        filename: str, Path, or None
            .blo filename to save.
            If None then the default file name is the same as the instance.
        binning: int
            Binning factor to apply to frames. Must divide frames evenly.
        clip: None or 2-tuple
            Intensity limits for each frame, eg. (0, 255).
        check_exists: bool
            If True then the file will not be created if a file with the same name already exists.
        signal: None or QSignal
            If provided the signal will emit on each iteration.
        kwds:
            Passed to tvips.get_blo_header.

        Notes
        -----
        Code has been recycled from two projects:
        [1] http://hyperspy.org/hyperspy-doc/current/index.html
        [2] https://pypi.org/project/tvipsconverter/

        """

        if filename is None:
            filename = self.files[0].with_suffix(".blo")

        filename = Path(filename)

        if check_exists and filename.exists():
            print(f"Filename: {str(filename)} already exists, aborting.")
            return

        assert filename.suffix.lower() == ".blo", "filename should have .blo extension."

        frame, frame_header = self.read_frame(0, return_header=True)

        # # test binning works
        try:
            bin_box(frame, binning, dtype=True)
        except Exception as err:
            raise Exception(f"During binning test the following error occurred: {err}")

        endianess = kwds.pop("endianess", "<")
        header, note = self.get_blo_header(
            binning=binning,
            endianess=endianess,
            Distortion_N01=1.0,
            Distortion_N09=1.0,
            Note="Reconstructed from TVIPS image stream",
            Camera_length=self.general_header["magtotal"],
            Beam_energy=self.general_header["ht"] * 1000,
            **kwds,
        )

        dtype = endianess + "u1"

        if clip is not None:
            assert (
                isinstance(clip, (list, tuple)) and len(clip) == 2
            ), "clip must be a 2-tuple of (min, max)."

        # relevant frame numbers
        n = list(range(self.scan_offset, math.prod(self.scan_shape) + self.scan_offset))

        # compute vbf and rescale before writing file
        vbf = img_as_ubyte(
            rescale_intensity(self.vbf_intensities[n], out_range=(0.0, 1.0))
        )

        # now start file write
        with open(filename, "wb") as f:
            # Write header
            header.tofile(f)
            # Write header note field:
            if len(note) > int(header["Data_offset_1"]) - f.tell():
                note = note[: int(header["Data_offset_1"]) - f.tell() - len(note)]
            f.write(note.encode())
            # Zero pad until next data block
            zero_pad = int(header["Data_offset_1"]) - f.tell()
            np.zeros((zero_pad,), np.byte).tofile(f)
            # Write virtual bright field
            vbf.tofile(f)
            # Zero pad until next data block

            if f.tell() > int(header["Data_offset_2"]):
                raise ValueError("Size does not match data dimensions.")
            zero_pad = int(header["Data_offset_2"]) - f.tell()
            np.zeros((zero_pad,), np.byte).tofile(f)

            # Write full data stack:
            # We need to pad each image with magic 'AA55', then a u32 serial
            dp_head = np.zeros(
                (1,), dtype=[("MAGIC", endianess + "u2"), ("ID", endianess + "u4")]
            )
            dp_head["MAGIC"] = 0x55AA
            # Write by loop:
            with warnings.catch_warnings():
                # suppress img_as_ubyte limit warnings
                warnings.simplefilter("ignore")

                for frame in self.read_frame(n, signal=signal):
                    frame = bin_box(frame, binning, dtype=True)
                    if clip is not None:
                        frame = np.clip(frame, *clip)
                    dp_head.tofile(f)
                    img_as_ubyte(frame).astype(dtype).tofile(f)
                    dp_head["ID"] += 1

        print(f"\nBlockfile written successfully: {filename}.")
