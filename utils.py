import numpy as np
from scipy import ndimage
from skimage import feature


def find_direct_beam(frame, side=30, sigma=3.0, refine=True, **kwargs):
    """

    Find the direct beam within a box from the center of the frame by blurring and getting the peak.

    Parameters
    ----------
    frame: ndarray
        The frame in question.
    side: int
        The side length of the box to find the central peak.
    sigma: float
        The Gaussian constant.
    refine: bool
        If True the direct beam position is further refined by its center of mass.
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


def recenter_mask(mask, new_center, old_center):
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


def roll_array_subpixel(arr, shift, mode="edge"):
    """
    Shift and array by a subpixel amount using local averaging.
    This method uses Manhattan distances and therefore retains local center of mass.

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
