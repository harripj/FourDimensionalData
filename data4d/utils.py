from itertools import product
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import ndimage
from skimage import feature


def bin_box(
    arr: NDArray,
    factor: int,
    axis: Optional[Union[int, Tuple[int, int]]] = None,
    dtype: Optional[DTypeLike] = False,
) -> NDArray:
    """
    Use box averaging to bin the images.

    Parameters
    ----------
    arr: ndarray
        Input array to box.
    factor: int
        The binning factor.
    axis: None, int, or tuple of ints
        Axis or axes to apply binning to.
        If None then all axes are binned.
    keep_dtype: bool or dtype
        If True then the output data type will be the same as arr.
        If False then the output type deafults to np.float.
        If dtype then this data type will be forced.


    Returns
    -------
    binned: ndarray
        The binned array.
    """
    if factor == 1:
        # no binning required
        return arr

    arr = np.asarray(arr)

    if axis is None:
        axis = tuple(range(arr.ndim))
    else:
        if isinstance(axis, (int, np.integer)):
            axes = (axis,)
        else:
            assert isinstance(
                axis, (list, tuple)
            ), "axes must be either int, list, or tuple."

    axis = tuple(a if a >= 0 else a + arr.ndim for a in axis)  # handle negative indices
    assert max(axis) <= arr.ndim, "axes must be within arr.ndim."
    assert all(
        isinstance(i, (int, np.integer)) for i in axis
    ), "All axes must be integers."

    assert all(
        not arr.shape[i] % factor for i in filter(lambda x: x is not None, axis)
    ), f"array shape is not factorisable by factor {factor}."

    # should work ndim
    slices = []
    for v in product(range(factor), repeat=len(axis)):
        # calculate all slicing offsets in all dimensions
        v = iter(v)
        temp = []
        for i in range(arr.ndim):
            # add slice object if axes is specified, otherwise no slicing
            temp.append(slice(next(v), None, factor) if i in axis else slice(None))
        slices.append(tuple(temp))

    # sort output data type
    if dtype is True:
        dtype = arr.dtype
    elif dtype is False:
        dtype = None
    # otherwise assume a valid data type

    # stack the offset slices and take mean down stack axis to finish binning
    return np.stack(tuple(arr[s] for s in slices), axis=0).mean(axis=0).astype(dtype)


def find_direct_beam(
    frame: NDArray,
    side: int = 30,
    sigma: float = 3.0,
    refine: bool = True,
    **kwargs: dict[str, Any],
) -> Tuple[int, int]:
    """
    Find the direct beam within a box from the center of the frame by
    blurring and getting the peak.

    Parameters
    ----------
    frame: ndarray
        The frame in question.
    side: int
        The side length of the box to find the central peak.
    sigma: float
        The Gaussian constant.
    refine: bool
        If True the direct beam position is further refined by its
        center of mass.
    kwargs:
        Passed to skimage.feature.peak_local_max.
        Defaults are exclude_border=1, threshold_rel=0.1.

    Returns
    -------
    ij: tuple of ints
        Frame center coordinates.

    """

    kwargs.setdefault("exclude_border", 1)
    kwargs.setdefault("threshold_rel", 0.1)

    def find_center(arr, sigma, refine=refine, mode="nearest", **kwargs):
        """
        Find central peak in array by blurring.
        Peak returned will be the one closest to the array center.
        """

        # get all peaks in blurred image (at least sigma away from each other)
        blurred = ndimage.gaussian_filter(arr, sigma, mode=mode)
        coords = feature.peak_local_max(blurred, int(sigma), **kwargs)

        if not coords.size:
            center = None
        # get peak closest to center
        else:
            center = coords[
                np.linalg.norm(coords - np.array(arr.shape) // 2, axis=1).argmin()
            ]
            if refine:
                # take a smaller crop of the original image and get center of mass
                arr_temp = arr[
                    tuple(
                        slice(c - 2 * int(sigma), c + 2 * int(sigma) + 1)
                        for c in center
                    )
                ]
                # then correct for crop offset
                center = tuple(
                    c - arr_temp.shape[i] // 2 + center[i]
                    for i, c in enumerate(ndimage.center_of_mass(arr_temp))
                )
        return center

    center = tuple(i // 2 for i in frame.shape)
    crop = frame[tuple(slice(c - side // 2, c + side // 2 + 1) for c in center)]
    # initial find center
    out = find_center(crop, sigma, **kwargs)

    if out is not None:
        out = tuple(o + c - side // 2 for o, c in zip(out, center))
    else:
        # try full image
        out = find_center(frame, sigma)
        if out is None:
            out = tuple(np.nan for i in frame.shape)

    return out


def recenter_mask(
    mask: ArrayLike, new_center: Tuple[int, int], old_center: Tuple[int, int]
) -> NDArray:
    """

    Shift a mask to a new center position.

    Parameters
    ----------
    mask: (M, N) ndarray or (N, ndim) tuple of arrays
        If mask is ndarray, it must be the same shape as frame. If tuple
        of arrays, these arrays represent coordinates within the frame.
        Note that if mask is ndarray the mask is just rolled through
        shift.
    new_center, old_center: iterable of ints
        The new and old center location of the mask.
        The shift is calculated as the difference between these two
        values. Must have the same length as mask.ndim.

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


def roll_array_subpixel(arr: NDArray, shift: ArrayLike, mode: str = "edge") -> NDArray:
    """
    Shift and array by a subpixel amount using local averaging.
    This method uses Manhattan distances and therefore retains local
    center of mass.

    Parameters
    ----------
    arr: (M, N, [P,]) ndarray
        Array to shift.
    shift: (arr.ndim,) array-like
        The shift values for each dimension.
    mode: str
        Passed to np.pad.

    Returns
    -------
    out: (M, N, [P,]) ndarray
        The shifted array.

    """
    shift_fractional = tuple(s % 1 for s in shift)

    # pad array to index
    pad = np.ceil(np.abs(shift)).astype(int)
    padded = np.pad(
        arr,
        tuple((0, p) if s > 0 else (p, 0) for s, p in zip(shift, pad)),
        mode=mode,
    )

    # get grid to index array
    grid = np.mgrid[tuple(slice(None, s) for s in arr.shape)]

    # need a local coordinates, ie every (0, 1) combination for every dimension
    cube = np.mgrid[tuple(slice(None, 2) for s in arr.shape)].T.reshape(-1, len(shift))

    # initialise outputs
    out = np.zeros_like(arr, dtype=float).ravel()
    weight_total = 0

    # loop over every index within local cube
    for indices in cube:
        # weight is 1/distance, Manhattan weighting
        weight = 1.0 / np.abs(indices - shift_fractional).prod()
        # if weight is inf, ie. 1/0, then change for no weighting
        weight = 0.0 if np.isinf(weight) else weight
        # keep track of total weighting
        weight_total += weight
        # index padded array and weight
        out += padded[tuple(i.ravel() for i in (grid.T + shift).T.astype(int))] * weight

    return (out / weight_total).reshape(arr.shape)
