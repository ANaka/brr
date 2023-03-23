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
