from dataclasses import dataclass
from pathlib import Path
from typing import Union
from ipywidgets.widgets import (
    interactive,
    Checkbox,
    FloatRangeSlider,
    IntSlider,
)
import numpy as np
from numpy.typing import DTypeLike
import math
from matplotlib import pyplot as plt


@dataclass
class FourDimensionalData:
    file: Union[str, Path]
    scan_shape: tuple
    frame_shape: tuple
    pixel_size_scan: tuple
    pixel_size_frame: tuple
    parameters: dict
    dtype: DTypeLike

    def __post_init__(self):
        self._scan_offset = 0  # in frames, used for TVIPS, for example

        self._frame_mean = None
        self._frame_max = None

    def read_frame(self, n):
        raise NotImplementedError("read_frame must be implemented by subclass.")

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
        """Total number of frames within dataset as calculated by the scan parameters."""
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

    def __getitem__(self, ij):
        if not isinstance(ij, (tuple, list, np.ndarray)) or not len(ij) == 2:
            raise ValueError("ij should be iterable of array of ints (i, j).")

        # spoof numpy slicing onto grid of coords and then extract coords
        grid = np.dstack(np.mgrid[: self.scan_y, : self.scan_x])
        # index grid of coords and get all ij coords to create n
        _ij = grid[ij]
        ij = np.stack((_ij[..., 0].ravel(), _ij[..., 1].ravel()))
        n = self.frame_index_from_ij(ij)

        shape_final = (
            _ij.shape[:-1] + self.frame_shape
        )  # just nav. axes of grid/_ij and frame
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

        _dtype = np.iinfo(self.dtype)
        _min, _max = _dtype.min, _dtype.max

        climrange = FloatRangeSlider(
            value=(_min, _max),
            min=_min,
            max=_max,
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
