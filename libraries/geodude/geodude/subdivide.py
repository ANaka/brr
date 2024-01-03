import numpy as np
from geodude.ops import get_bottom, get_left, get_right, get_top
from geodude.parameter import Prm
from geodude.utils import merge_Polygons
from shapely.geometry import LineString, box
from shapely.ops import nearest_points


def quadrant_split(square_poly):
    upper_left = box(
        minx=get_left(square_poly),
        miny=square_poly.centroid.y,
        maxx=square_poly.centroid.x,
        maxy=get_top(square_poly),
        ccw=True,
    )
    upper_right = box(
        minx=square_poly.centroid.x,
        miny=square_poly.centroid.y,
        maxx=get_right(square_poly),
        maxy=get_top(square_poly),
        ccw=True,
    )
    lower_right = box(
        minx=square_poly.centroid.x,
        miny=get_bottom(square_poly),
        maxx=get_right(square_poly),
        maxy=square_poly.centroid.y,
        ccw=True,
    )
    lower_left = box(
        minx=get_left(square_poly),
        miny=get_bottom(square_poly),
        maxx=square_poly.centroid.x,
        maxy=square_poly.centroid.y,
        ccw=True,
    )
    quadrants = merge_Polygons([upper_left, upper_right, lower_right, lower_left])
    return quadrants


def permit_by_index(i, p, allowed_indices):
    # whats up w the ununsed vars?
    return i in allowed_indices


def bino_draw(i, p, p_continue=1):
    return np.random.binomial(n=1, p=p_continue)


def permit_all(i, p):
    return True


def flex_rule_recursive_split(poly, split_func, continue_func, depth=0, depth_limit=15, buffer_kwargs=None):

    if buffer_kwargs is None:
        buffer_kwargs = {"distance": 0}
    polys = split_func(poly)
    split_polys = []
    for i, p in enumerate(polys.geoms):
        continue_draw = continue_func(i=i, p=p)

        if continue_draw and (depth < depth_limit):

            split_polys += flex_rule_recursive_split(
                p,
                split_func=split_func,
                continue_func=continue_func,
                depth=depth + 1,
                depth_limit=depth_limit,
                buffer_kwargs=buffer_kwargs,
            )
        else:
            split_polys.append(p.buffer(**buffer_kwargs))
    return split_polys


class ContinuePolicy(object):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, i, p, depth, poly):
        return self.policy(i, p, depth, poly)


def even_odd(i, p, depth, poly):
    lookup = {0: [0, 2], 1: [1, 3]}
    mod = depth % 2
    return i in lookup.get(mod)


def mod4(i, p, depth, poly):
    lookup = {0: [0], 1: [1], 2: [2], 3: [3]}
    mod = depth % 4
    return i in lookup.get(mod)


def permit_by_index_depth_dependent(i, p, allowed_index_lookup):
    return


def very_flex_rule_recursive_split(poly, split_func, continue_func, depth=0, depth_limit=15, buffer_kwargs=None):

    if buffer_kwargs is None:
        buffer_kwargs = {"distance": 0}
    polys = split_func(poly)
    split_polys = []
    for i, p in enumerate(polys):
        continue_draw = continue_func(i, p, depth, poly)

        if continue_draw and (depth < depth_limit):

            split_polys += very_flex_rule_recursive_split(
                p,
                split_func=split_func,
                continue_func=continue_func,
                depth=depth + 1,
                depth_limit=depth_limit,
                buffer_kwargs=buffer_kwargs,
            )
        else:
            split_polys.append(p.buffer(**buffer_kwargs))
    return split_polys


def distance_from_pt(i, p, depth, poly, target, p_range, d_range):
    d = poly.distance(target)
    p_continue = np.interp(d, d_range, p_range)
    return np.random.binomial(n=1, p=p_continue)


# splitting line funcs
def random_line_subdivide_gen(poly, x0gen=None, x1gen=None):
    x0 = x0gen()
    x1 = x1gen()
    return LineString([poly.boundary.interpolate(x, normalized=True) for x in [x0, x1]])


def random_line_subdivide(poly, x0=None, x1=None):
    if x0 is None:
        x0 = np.random.uniform(0, 1)
    if x1 is None:
        x1 = (x0 + 0.5) % 1
    return LineString([poly.boundary.interpolate(x, normalized=True) for x in [x0, x1]])


# def random_bezier_subdivide(poly, x0=None, x1=None, n_eval_points=50):
#     if x0 is None:
#         x0 = np.random.uniform(0.2, 0.4)
#     if x1 is None:
#         x1 = np.random.uniform(0.6, 0.8)
#     line = np.asfortranarray(random_line_subdivide(poly, x0, x1))
#     bez_array = np.stack([line[0], poly.centroid, line[1]]).T
#     curve1 = bezier.Curve(bez_array, degree=2)
#     bez = curve1.evaluate_multi(np.linspace(0.0, 1.0, n_eval_points))
#     return sg.asLineString(bez.T)


def longest_side_of_min_rectangle_subdivide(poly, xgen=None, mirrored: bool = True):
    if xgen is None:
        xgen = Prm(0.5)
    mrrc = poly.minimum_rotated_rectangle.boundary.coords
    sides = [LineString([mrrc[i], mrrc[i + 1]]) for i in range(4)]
    longest_sides = [sides[i] for i in np.argsort([-ls.length for ls in sides])[:2]]
    if mirrored:
        longest_sides[1] = LineString([pt for pt in reversed(longest_sides[1].coords)])
    bps = [ls.interpolate(xgen(), normalized=True) for ls in longest_sides]

    return LineString([nearest_points(bp, poly.boundary)[1] for bp in bps])


def split_poly(poly, line):
    return merge_Polygons(poly.difference(line.buffer(1e-6)))


def split_along_longest_side_of_min_rectangle(poly, xgen=None):
    line = longest_side_of_min_rectangle_subdivide(poly=poly, xgen=xgen)
    return merge_Polygons(split_poly(poly, line))


# def split_random_bezier(
#     poly,
#     x0=None,
#     x1=None,
#     n_eval_points=50,
# ):
#     line = random_bezier_subdivide(poly, x0=x0, x1=x1, n_eval_points=n_eval_points)
#     return merge_Polygons(split_poly(poly, line))


def split_random_line_gen(poly, x0gen=None, x1gen=None):
    line = random_line_subdivide_gen(poly, x0gen=x0gen, x1gen=x1gen)
    return merge_Polygons(split_poly(poly, line))
