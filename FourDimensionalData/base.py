from __future__ import annotations

import abc
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Union

from ipywidgets.widgets import Checkbox, FloatRangeSlider, IntSlider, interactive
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import DTypeLike
from skimage import draw

from .utils import find_direct_beam, recenter_mask


@dataclass
class FourDimensionalData(abc.ABC):
    file: Union[str, Path]
    scan_shape: tuple
    frame_shape: tuple
    pixel_size_scan: tuple
    pixel_size_frame: tuple
    parameters: dict
    dtype: DTypeLike

    def __post_init__(self):
        self.file = Path(self.file)
        self._scan_offset = 0  # in frames, used for TVIPS, for example
        self._frame_mean = None
        self._frame_max = None
        self._frame_total = None
        self._vbf_intensities = None
        self._direct_beam_coordinates = np.full(
            (len(self), len(self.frame_shape)), -1
        )  # -1 meaning not calculated

    @abc.abstractmethod
    def read_frame(self, n, signal=None):
        raise NotImplementedError("read_frame must be implemented by subclass.")

    @property
    def data(self):
        return self.read_frame(None)

    def frame_index_from_ij(self, ij):
        """Return frame number from scan coordinate."""
        return np.ravel_multi_index(ij, self.scan_shape) + self.scan_offset

    def __len__(self):
        return self.number_of_frames

    @property
    def scan_offset(self):
        """Number of frames before start of scan, may be 0."""
        return self._scan_offset

    @scan_offset.setter
    def scan_offset(self, x):
        self._scan_offset = x

    @property
    def shape(self):
        """4D shape of dataset."""
        return self.scan_shape + self.frame_shape

    @property
    def scan_x(self):
        """The number of frames in the scan x-dimension."""
        return self.scan_shape[1]

    @scan_x.setter
    def scan_x(self, x):
        self.scan_shape = (self.scan_y, x)

    @property
    def scan_y(self):
        """The number of frames in the scan y-dimension."""
        return self.scan_shape[0]

    @scan_y.setter
    def scan_y(self, y):
        self.scan_shape = (y, self.scan_x)

    def check_scan_shape(self):
        """Check whether scan shape (x, y) with offset fits into file."""
        return (
            True
            if self.scan_offset + self.scan_x * self.scan_y <= self.number_of_frames
            else False
        )

    @property
    def number_of_frames(self):
        """Total number of frames within dataset as calculated by the
        scan parameters."""
        return math.prod(self.shape) + self.scan_offset

    @property
    def frame_max(self):
        """Calculate or return diffraction pattern max."""
        if self._frame_max is None:
            for i, f in enumerate(self.read_frame(None)):
                if not i:
                    frame_max = f
                else:
                    frame_max = np.stack((frame_max, f)).max(axis=0)
            self._frame_max = frame_max

        return self._frame_max

    @property
    def frame_mean(self):
        """Calculate or return diffraction pattern mean."""
        if self._frame_mean is None:
            for i, f in enumerate(self.read_frame(None)):
                if not i:
                    frame_mean = f
                else:
                    frame_mean = np.stack((frame_mean, f)).mean(
                        axis=0, dtype=frame_mean.dtype
                    )
            self._frame_mean = frame_mean

        return self._frame_mean

    def _calculate_frame_total(self):
        total = np.empty(self.number_of_frames, dtype=float)
        for i, f in enumerate(self.read_frame(None)):
            total[i] = f.sum()
        return total

    @property
    def frame_total(self):
        """Calculate or return diffraction pattern total."""
        if self._frame_total is None:
            total = self._calculate_frame_total()
            self._frame_total = total
        return self._frame_total

    @property
    def vbf_intensities(self):
        return self._vbf_intensities

    @vbf_intensities.setter
    def vbf_intensities(self, x):
        x = np.asarray(x)
        assert x.size == len(
            self
        ), f"VBF intensities with shape: {x.size} are not defined for every frame: {len(self)}."
        self._vbf_intensities = np.asarray(x)

    @property
    def direct_beam_coordinates(self):
        return self._direct_beam_coordinates

    @direct_beam_coordinates.setter
    def direct_beam_coordinates(self, c):
        assert len(c) == len(
            self
        ), f"Cooridnates with shape: {len(c)} are not defined for every frame: {len(self)}."
        c = np.asarray(c)
        assert c.shape[-1] == len(
            self.frame_shape
        ), f"Coordinates with shape: {c.shape} should be defined for each frame dimension: {len(self.frame_shape)}."
        self._direct_beam_coordinates = c

    def calculate_virtual_bright_field_reconstruction(
        self,
        radius=5,
        n=None,
        recenter=False,
        side=30,
        sigma=3.0,
    ):
        """
        Calculate virtual bright field reconstruction, accessible at
        self.vbf_intensities.

        Parameters
        ----------
        radius: float
            The VBF aperture radius in pixels.
        n: None or array-like
            Frame numbers to compute.
            If None then all frames are computed.
        recenter: bool
            If True then the direct beam position is computed and the
            mask is recentered on this location. It is assumed that mask
            is initially centered on the center of each frame.
        side: int
            Box side size for computing the central beam location.
        sigma: float
            Gaussian parameter for the recentering algorithm.

        """

        # compute vbf
        center = (i / 2.0 for i in self.frame_shape)
        coords = draw.disk(center, radius=radius, shape=self.frame_shape)

        self.vbf_intensities = self.calculate_virtual_reconstruction(
            coords, n, recenter=recenter, side=side, sigma=sigma
        )

    def calculate_virtual_reconstruction(
        self, mask, n=None, recenter=False, integrate=True, side=50, sigma=3.0
    ):
        """

        Calculate virtual reconstruction from a mask.

        Parameters
        ----------
        mask: (M, N) ndarray or (N, ndim) tuple of arrays
            If mask is ndarray, it must be the same shape as frame.
            If tuple or arrays, these arrays represent coordinates
            within the frame.
        n: None or array-like
            Frame numbers to compute.
            If None then all frames are computed.
        recenter: bool
            If True then the direct beam position is computed and the
            mask is recentered on this location. It is assumed that mask
            is initially centered on the center of each frame.
        integrate: bool
            If True then the extracted intensity from each point in mask
            is integrated (summed). If False then the extracted
            intensities from each point in mask are returned as an
            array.
        side: int
            Box side size for computing the central beam location.
        sigma: float
            Gaussian parameter for the recentering algorithm.

        Returns
        -------
        intensities: (n,) ndarray-like
            The reconstructed intensities.

        """
        # check inputs
        if isinstance(mask, np.ndarray):
            assert (
                mask.shape == self.frame_shape
            ), f"mask shape: {mask.shape} does not equal frame shape: {self.frame_shape}."
        elif isinstance(mask, tuple):
            assert len(mask) == len(
                self.frame_shape
            ), f"mask has incorrect dimensionality: {len(mask)}."
        else:
            raise ValueError("mask must be ndarray or tuple of integer arrays.")

        # get direct beam coords to avoid re-reads
        dbc = self.direct_beam_coordinates
        # init output
        intensities = []

        for frame, num in self.read_frame(n, return_index=True):
            if recenter:
                # if the data has not been written, ie. -1
                if (dbc[num] < 0).any():
                    center = find_direct_beam(frame, side, sigma)
                    # update center_location and unmask array
                    dbc[num] = center
                # otherwise read
                else:
                    center = dbc[num]

                mask_temp = recenter_mask(
                    mask, center, tuple(i // 2 for i in frame.shape)
                )
            else:
                mask_temp = mask

            # extract intensities from frame
            intensities_temp = frame[mask_temp]
            if integrate:
                intensities_temp = intensities_temp.sum()
            intensities.append(intensities_temp)

        if recenter:
            # write to file after using all frames
            self.direct_beam_coordinates = dbc

        return intensities

    def __repr__(self):
        return f"{self.__class__.__name__} {self.shape} {self.file.stem}"

    def __getitem__(self, ij):
        if not isinstance(ij, (tuple, list, np.ndarray)) or not len(ij) == 2:
            raise ValueError("ij should be iterable of array of ints (i, j).")

        # calculate (i, j, 2) grid of scan coords useful for slicing
        scan_grid = np.dstack(np.mgrid[: self.scan_y, : self.scan_x])
        # index grid of coords and get all ij coords to create n
        ij_indexed = scan_grid[ij]
        n = self.frame_index_from_ij(
            np.stack((ij_indexed[..., 0].ravel(), ij_indexed[..., 1].ravel()))
        )

        shape_final = (
            ij_indexed.shape[:-1] + self.frame_shape
        )  # axes of ij_indexed and frame
        return np.reshape(tuple(self.read_frame(n)), shape_final).squeeze()

    def plot(self, figsize=(12, 6), cmap="gray"):
        """Produce interactive figure to show data. Returns the widget."""
        fig, ax = plt.subplots(ncols=2, figsize=figsize)

        # initial images
        vbf_image = ax[0].matshow(self.vbf, cmap=cmap)

        # plot initial image
        i, j = 0, 0
        frame = self[i, j]
        diffraction_image = ax[1].matshow(frame, cmap=cmap)

        crosshair_vertical = ax[0].axvline(j, color="k", alpha=0.4)
        crosshair_horizontal = ax[0].axhline(i, color="k", alpha=0.4)

        # define some useful widgets
        i = IntSlider(i, 0, self.scan_shape[0] - 1, description="i")
        j = IntSlider(j, 0, self.scan_shape[1] - 1, description="j")
        log = Checkbox(True)
        clim = Checkbox(True, description="Auto clim?")

        dtype = np.iinfo(self.dtype)
        min, max = dtype.min, dtype.max

        climrange = FloatRangeSlider(
            value=(min, max),
            min=min,
            max=max,
            description="Manual limits:",
            orientation="horizontal",
        )

        def update(ii, jj, log, clim, climrange):
            # change slider limits to avoid over reading data
            crosshair_vertical.set_xdata((jj, jj))
            crosshair_horizontal.set_ydata((ii, ii))

            diffraction = self.read_frame(self.frame_index_from_ij((ii, jj)))
            if log:
                diffraction = np.log(
                    diffraction.astype(np.float32) - diffraction.min() + 1
                )
            diffraction_image.set_array(diffraction)

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

        cid = []  # not sure whether storing these in a local list does anything...
        cid.append(fig.canvas.mpl_connect("button_press_event", mouse_clicked))
        cid.append(fig.canvas.mpl_connect("button_release_event", mouse_released))
        cid.append(fig.canvas.mpl_connect("figure_enter_event", axes_enter))
        cid.append(fig.canvas.mpl_connect("figure_leave_event", axes_leave))
        cid.append(fig.canvas.mpl_connect("motion_notify_event", mouse_dragged))

        return interactive(
            update,
            ii=i,
            jj=j,
            log=log,
            clim=clim,
            climrange=climrange,
        )

    @staticmethod
    def from_file(fname: Union[str, Path]) -> FourDimensionalData:
        """Open a 4D data file with the appropriate reader.

        Currently supported formats are ASTAR .blo and TVIPS .tvips.
        """
        from .TVIPS import TVIPS
        from .blockfile import BLO

        handler = {"blo": BLO, "tvips": TVIPS}

        fname = Path(fname)
        ext = fname.suffix.strip(os.extsep)

        if ext not in handler:
            raise TypeError(
                f"File format not supported. Supported formats are: {handler.keys()}."
            )
        return handler[ext](fname)
