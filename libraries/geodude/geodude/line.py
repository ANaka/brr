import numpy as np
from shapely import LineString, MultiLineString


def bezier_func(points):
    """
    Create a Bezier curve function given a list of control points.

    Arguments:
    points -- a list of tuples representing the (x, y) coordinates of the control points

    Returns:
    A function that takes a 1D array of values t between 0 and 1 and returns a 2D array of (x, y) coordinates on the curve.
    """
    # Define the De Casteljau's algorithm to recursively subdivide the control points
    def de_casteljau(t, points):
        if len(points) == 1:
            return points[0]
        else:
            x_coords, y_coords = zip(*points)
            new_x_coords = (1 - t) * np.array(x_coords[:-1]) + t * np.array(x_coords[1:])
            new_y_coords = (1 - t) * np.array(y_coords[:-1]) + t * np.array(y_coords[1:])
            new_points = list(zip(new_x_coords, new_y_coords))
            return de_casteljau(t, new_points)

    # Define the Bezier curve function that calls the De Casteljau's algorithm
    def evaluate_bezier(t, as_numpy: bool = False):
        """
        Evaluate the Bezier curve at a point t between 0 and 1.

        Arguments:
        t -- a 1D array of values between 0 and 1

        Returns:
        A 2D array of (x, y) coordinates on the curve at the given values of t.
        """
        pts = np.array([de_casteljau(ti, points) for ti in t])
        if as_numpy:
            return pts
        else:
            return LineString(pts)

    return evaluate_bezier


def dash_linestring(linestring, interpolation_distances):
    new_lines = []
    ii = 0
    while ii < len(interpolation_distances) - 1:
        pt0 = linestring.interpolate(interpolation_distances[ii], normalized=True)
        pt1 = linestring.interpolate(interpolation_distances[ii + 1], normalized=True)
        new_line = LineString([pt0, pt1])
        new_lines.append(new_line)
        ii += 2
    return MultiLineString(new_lines)
