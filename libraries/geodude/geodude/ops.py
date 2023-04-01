from dataclasses import dataclass, field

import numpy as np
from makefun import wraps
from shapely import Geometry
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
    **kwargs
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


def multi_buffer_translate(
    p, d_buffers, d_translates, angles, cap_style=2, join_style=2, return_original=True, **kwargs
):

    d_translates = ensure_collection(d_translates, length=len(d_buffers))
    angles = ensure_collection(angles, length=len(d_buffers))

    ssps = []
    if return_original:
        ssps.append(p)

    ssp = p
    for d_buffer, d_translate, angle in zip(d_buffers, d_translates, angles):
        ssp = buffer_translate(ssp, d_buffer, d_translate, angle, cap_style, join_style, **kwargs)
        if ssp.area < np.finfo(float).eps:
            break
        ssps.append(ssp)
    return ssps


@dataclass
class MultiBufferTranslatePrms:
    """
    pt = Point(0,0)
    poly = Poly(pt.buffer(10))
    prms = ScaleTransPrms(
        n_iters=200,
        d_buffer=-0.25,
        d_translate_factor=0.7,
        angles=0,
    )
    poly.fill_scale_trans(**prms.prms)
    """

    n_iters: int = 100
    d_buffer: float = -0.25
    d_translate_factor: float = 0.9
    d_translate: float = None
    angles: float = 0.0  # radians
    d_translates: list = field(default=None, init=False)

    def __post_init__(self):
        self.d_buffers = np.array([self.d_buffer] * self.n_iters)

        if self.d_translates is None:
            if self.d_translate is not None:
                self.d_translates = np.array([self.d_translate] * self.n_iters)
            else:
                self.d_translates = self.d_buffers * self.d_translate_factor

    @property
    def prms(self):
        varnames = ["d_buffers", "d_translates", "angles"]
        return {var: getattr(self, var) for var in varnames}
