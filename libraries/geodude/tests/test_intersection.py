import pytest
from geodude.intersection import (
    find_intersecting_polys,
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
