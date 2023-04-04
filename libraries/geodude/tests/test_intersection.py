import pytest
from geodude.intersection import (
    find_intersecting_polys,
    find_touching_polys,
    pairwise_partition_polygons,
    polys_to_gdf,
)
from shapely.geometry import Polygon
from shapely.ops import unary_union


def test_pairwise_partition_polygons():
    # Test input data
    r1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    r2 = Polygon([(5, 5), (5, 20), (15, 20), (15, 5)])
    r3 = Polygon([(10, 0), (10, 10), (20, 10), (20, 0)])

    geoms = [r1, r2, r3]
    gdf = polys_to_gdf(geoms)

    disjoint, n_intersections_log = pairwise_partition_polygons(gdf)

    # Expected results
    expected_disjoint_length = 5

    # Check if the output matches the expected results
    assert len(disjoint) == expected_disjoint_length

    # Check if the output geometries cover the same area as the input geometries
    input_area = unary_union(geoms).area
    output_area = disjoint["geometry"].apply(lambda x: x.area).sum()
    assert pytest.approx(input_area, rel=1e-6) == output_area

    # Check if the union of the output geometries is equal to the union of the input geometries
    assert unary_union(disjoint["geometry"].tolist()).equals(unary_union(geoms))

    # Check if the output geometries are disjoint
    intersection_map = find_intersecting_polys(disjoint.geometry)
    assert sum(len(x) for x in intersection_map.values()) == 0


class TestFindIntersectingPolys:

    # Tests that the function correctly identifies identical polygons. tags: [happy path]
    def test_identical_geoms(self):
        # Happy path test for identical polygons
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly_list = [poly1, poly2]
        result = find_intersecting_polys(poly_list)
        assert result == {0: {1}, 1: {0}}

        # Tests that the function correctly handles polygons that do not intersect with any other polygon. tags: [edge case]

    def test_non_intersecting_geoms(self):
        # Test that the function correctly handles polygons that do not intersect with any other polygon
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        poly3 = Polygon([(4, 4), (4, 5), (5, 5), (5, 4)])
        geoms = [poly1, poly2, poly3]
        intersection_map = find_intersecting_polys(geoms)
        assert intersection_map == {0: set(), 1: set(), 2: set()}

        # Tests that the function correctly handles polygons that intersect with all other polygons. tags: [edge case]

    def test_all_intersecting_geoms(self):
        # Test that the function correctly handles polygons that intersect with all other polygons
        poly1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        poly2 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1)])
        poly3 = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
        geoms = [poly1, poly2, poly3]
        intersection_map = find_intersecting_polys(geoms)
        assert intersection_map == {
            0: {
                1,
            },
            1: {0, 2},
            2: {1},
        }


class TestFindTouchingPolys:

    # Tests that the function correctly identifies touching polygons when geoms contains multiple polygons and vectorized is true. tags: [happy path]
    def test_multiple_polygons_vectorized(self):
        # Happy path test for multiple polygons with vectorized=True
        geoms = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]),
            Polygon([(2, 2), (2, 3), (3, 3), (3, 2)]),
        ]
        result = find_touching_polys(geoms, vectorized=True)
        assert result == {0: {1}, 1: {0, 2}, 2: {1}}

        # Tests that the function correctly identifies non-touching polygons when geoms contains only one polygon and vectorized is false.

    def test_single_polygon_non_vectorized(self):
        # Happy path test for single polygon with vectorized=False
        geoms = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        result = find_touching_polys(geoms, vectorized=False)
        assert result == {0: set()}

        # Tests that the function correctly identifies non-touching polygons when geoms contains polygons with no intersection. tags: [edge case]

    def test_no_intersection(self):
        # Test that the function correctly identifies non-touching polygons when geoms contains polygons with no intersection
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        poly3 = Polygon([(4, 4), (4, 5), (5, 5), (5, 4)])
        geoms = [poly1, poly2, poly3]
        result = find_touching_polys(geoms)
        assert result == {0: set(), 1: set(), 2: set()}
