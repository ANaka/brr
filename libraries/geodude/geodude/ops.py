from typing import Union

import numpy as np
from geodude.parameter import Prm, unpack_prms
from makefun import wraps
from shapely import Geometry, MultiPolygon
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
    return MultiPolygon(geoms)
