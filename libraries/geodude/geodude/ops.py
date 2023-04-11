from functools import partial, singledispatch
from typing import Union

import numpy as np
from geodude.parameter import Prm, unpack_prms
from geodude.utils import merge_Polygons
from makefun import wraps
from shapely import Geometry, LineString
from shapely import affinity as sa
from shapely.geometry import box


def get_left(geom):
    return geom.bounds[0]


def get_bottom(geom):
    return geom.bounds[1]


def get_right(geom):
    return geom.bounds[2]


def get_top(geom):
    return geom.bounds[3]


def get_width(geom):
    return get_right(geom) - get_left(geom)


def get_height(geom):
    return get_top(geom) - get_bottom(geom)


def center_at(geom, target_point, return_transform=False, use_centroid=False):
    transform = {}
    if use_centroid:
        init_point = geom.centroid
    else:
        init_point = box(*geom.bounds).centroid
    transform["xoff"] = target_point.x - init_point.x
    transform["yoff"] = target_point.y - init_point.y

    translated = sa.translate(geom, **transform)
    if return_transform:
        return translated, transform
    else:
        return translated


def scale_like(geom, target_geom, preserve_aspect_ratio=True, use_smaller=True, return_transform=False):
    width_g = get_width(geom)
    height_g = get_height(geom)

    width_tg = get_width(target_geom)
    height_tg = get_height(target_geom)

    xfact = width_tg / width_g
    yfact = height_tg / height_g
    if preserve_aspect_ratio and use_smaller:
        fact = min([xfact, yfact])
        transform = {"xfact": fact, "yfact": fact}
        scaled = sa.scale(geom, **transform)
    elif preserve_aspect_ratio and not use_smaller:
        fact = max([xfact, yfact])
        transform = {"xfact": fact, "yfact": fact}
        scaled = sa.scale(geom, **transform)
    else:
        transform = {"xfact": xfact, "yfact": yfact}
        scaled = sa.scale(geom, **transform)

    if return_transform:
        return scaled, transform
    else:
        return scaled


def make_like(p, target, return_transform=False):
    "rescale and center, good for making it fit in a drawbox"
    scaled, transform = scale_like(p, target, return_transform=True)
    transformed_poly, translate_transform = center_at(scaled, target.centroid, return_transform=True)

    transform.update(translate_transform)
    if return_transform:
        return transformed_poly, transform
    else:
        return transformed_poly


def scalar_to_collection(scalar, length):
    stype = type(scalar)
    return (np.ones(length) * scalar).astype(stype)


def ensure_collection(x, length):
    if np.iterable(x):
        assert len(x) == length
        return x
    else:
        return scalar_to_collection(x, length)


def angle_translate(geom: Geometry, d_translate: float, deg: float = None, rad: float = None, **kwargs) -> Geometry:
    assert (deg is None) ^ (rad is None), "must specify either deg or rad"
    if deg is not None:
        rad = np.deg2rad(deg)
    xoff = np.cos(rad) * d_translate
    yoff = np.sin(rad) * d_translate
    return sa.translate(geom, xoff=xoff, yoff=yoff, **kwargs)


@unpack_prms
def buffer_translate(
    geom: Geometry,
    d_buffer: float,
    d_translate: float,
    deg: float = None,
    rad: float = None,
    resolution=16,
    quad_segs=8,
    cap_style="round",
    join_style="round",
    mitre_limit=5.0,
    single_sided=False,
    **kwargs,
):
    bp = geom.buffer(
        distance=d_buffer,
        cap_style=cap_style,
        join_style=join_style,
        resolution=resolution,
        quad_segs=quad_segs,
        mitre_limit=mitre_limit,
        single_sided=single_sided,
    )
    return angle_translate(bp, d_translate, deg=deg, rad=rad, **kwargs)


@wraps(buffer_translate)
def buft(*args, **kwargs):
    return buffer_translate(*args, **kwargs)


def buft_fill(
    geom,
    d_buffer: Union[float, Prm] = None,
    d_translate: Union[float, Prm] = None,
    deg: Union[float, Prm] = None,
    rad: Union[float, Prm] = None,
    n_iters: int = 200,
    include_original: bool = True,
    **kwargs,
):

    if d_buffer is None:
        d_buffer = Prm(-0.9)

    if d_translate is None:
        d_translate = Prm(lambda: Prm(d_buffer)() * 0.9)

    geoms = []
    if include_original:
        geoms.append(geom)
    for ii in range(n_iters):
        geom = buft(
            geom=geom,
            d_buffer=d_buffer,
            d_translate=d_translate,
            deg=deg,
            rad=rad,
            **kwargs,
        )
        if geom.area < np.finfo(float).eps:
            break
        geoms.append(geom)
    return merge_Polygons(geoms)


# Define the base function for forming an orthonormal basis
@singledispatch
def form_orthonormal_basis(vec):
    raise NotImplementedError("Input type not supported")


# Register the specialized function for numpy.ndarray input type
@form_orthonormal_basis.register(np.ndarray)
def _(vec):
    # Normalize the input vector
    vec = vec / np.linalg.norm(vec)

    # Generate an orthogonal vector
    if vec[1] == 0:
        orthogonal_vec = np.array([0, 1])
    else:
        orthogonal_vec = np.array([-vec[1], vec[0]])

    # Normalize the orthogonal vector
    orthogonal_vec = orthogonal_vec / np.linalg.norm(orthogonal_vec)

    # Form the orthonormal basis matrix
    basis_matrix = np.column_stack((vec, orthogonal_vec))

    # Compute the inverse of the basis matrix
    inverse_basis_matrix = np.linalg.inv(basis_matrix)

    return basis_matrix, inverse_basis_matrix


# Register the specialized function for shapely.geometry.LineString input type
@form_orthonormal_basis.register(LineString)
def _(line):
    # Extract the start and end points of the LineString
    start, end = line.coords
    # Compute the vector representing the LineString
    vec = np.array(end) - np.array(start)
    # Call the function for numpy.ndarray input type to compute the orthonormal basis
    return form_orthonormal_basis(vec)


def form_affine_basis(arr: Union[np.ndarray, tuple, list, LineString], longest_dim_first: bool = True):
    """
    Computes and returns the affine basis matrix and its inverse based on the input array.

    Parameters
    ----------
    arr : Union[np.ndarray, tuple, list, LineString]
        The input array, which can be of type np.ndarray, tuple, list, or LineString.
        Must have shape (2,) or (2, 2). If the input array has shape (2,), the origin
        is prepended to form a 2x2 array.

    longest_dim_first : bool, optional, default: False
        If True, ensures that the longest dimension of the input vector is mapped onto
        the first dimension of the new basis. If False, the basis is formed without
        considering the longest dimension.

    Returns
    -------
    affine_basis_matrix : np.ndarray
        The 3x3 affine basis matrix, including scaling, rotation, and translation components.

    affine_inverse_basis_matrix : np.ndarray
        The 3x3 inverse of the affine basis matrix.

    Raises
    ------
    ValueError
        If the input array does not have shape (2, 2) or (2,).

    Example
    -------
    >>> arr = np.array([[0, 0], [2, 3]])
    >>> A, A_inv = form_affine_basis(arr, longest_dim_first=True)
    """
    # cast to numpy
    if isinstance(arr, LineString):
        arr = np.array(arr.coords)
    else:
        arr = np.array(arr)

    # If the input array has shape=(2,), prepend the origin to form a 2x2 array
    if arr.shape == (2,):
        arr = np.vstack(([0, 0], arr))

    # Ensure that the input array has shape=(2,2)
    if arr.shape != (2, 2):
        raise ValueError(f"Input array must have shape=(2,2) or shape=(2,), but shape is {arr.shape}")

    # Extract the start and end points (translation components and vector)
    start, vec = arr

    # Calculate the rotation angle based on the longest_dim_first parameter
    if longest_dim_first:
        if abs(vec[0]) >= abs(vec[1]):
            # swapdims
            start = start[::-1]
            vec = vec[::-1]

    tx, ty = start
    angle = np.arctan2(vec[1], vec[0])

    # Generate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # Calculate scaling
    scale = np.linalg.norm(vec)
    scaling_matrix = np.array([[scale, 0], [0, scale]])

    # Compute the inverse rotation matrix and inverse scaling matrix
    inverse_rotation_matrix = rotation_matrix.T
    inverse_scaling_matrix = np.linalg.inv(scaling_matrix)

    # Convert the 2x2 matrices to 3x3 affine transformation matrices
    affine_basis_matrix = np.eye(3)
    affine_basis_matrix[:2, :2] = rotation_matrix @ scaling_matrix
    affine_basis_matrix[:2, 2] = [tx, ty]  # Add translation components

    affine_inverse_basis_matrix = np.eye(3)
    affine_inverse_basis_matrix[:2, :2] = inverse_scaling_matrix @ inverse_rotation_matrix
    affine_inverse_basis_matrix[:2, 2] = (
        -inverse_rotation_matrix @ inverse_scaling_matrix @ [tx, ty]
    )  # Inverse translation components

    return affine_basis_matrix, affine_inverse_basis_matrix


def get_affine_transformation(arr: Union[LineString, np.ndarray], longest_dim_first: bool = True):
    """Return a partial function that takes a shapely.geometry object and applies the affine
    transformation defined by the input array. The input array must have shape=(2,) or (2,2).
    If the input array has shape=(2,), the origin is prepended to form a 2x2 array.

    from shapely import LineString
    from geodude.ops import get_affine_transformation
    from geodude.line import bezier_func
    line = LineString(((1, 0), (3, 1)))
    f = get_affine_transformation(line) # af is a partial function
    g = bezier_func([(0., 0), (.25, 2), (.5, -2), (.75, 3), (1, 0)])

    x = np.linspace(0, 1, 10)
    bez = g(x)
    transformed_bez = f(bez)  # i.e. f(g(x))
    """
    A, A_inv = form_affine_basis(arr, longest_dim_first=longest_dim_first)
    coefs = np.concatenate((A[0, :2], A[1, :2], A[:2, 2]))
    return partial(sa.affine_transform, matrix=coefs)
