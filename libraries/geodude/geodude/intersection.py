import itertools
from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union


def polys_to_gdf(polygons: Union[List, MultiPolygon]):
    try:
        if polygons.geom_type == "MultiPolygon":
            polygons = polygons.geoms
    except AttributeError:
        pass

    # Create a GeoDataFrame from the input polygons with an additional column to store the index
    return gpd.GeoDataFrame({"geometry": polygons})


def find_intersecting_polys(geoms: list):
    intersection_map = {ii: set() for ii in range(len(geoms))}
    for ii, jj in itertools.combinations(range(len(geoms)), 2):
        if geoms[ii].buffer(-1e-6).intersects(geoms[jj].buffer(-1e-6)):
            intersection_map[ii].add(jj)
            intersection_map[jj].add(ii)
    return intersection_map


def find_touching_polys(geoms: list):
    intersection_map = {ii: set() for ii in range(len(geoms))}
    for ii, jj in itertools.combinations(range(len(geoms)), 2):
        try:
            if geoms[ii].touches(geoms[jj]):
                intersection_map[ii].add(jj)
                intersection_map[jj].add(ii)
        except GEOSException:
            if geoms[ii].buffer(1e-6).intersects(geoms.buffer(1e-6)[jj]) and not geoms[ii].buffer(-1e-6).intersects(
                geoms[jj].buffer(-1e-6)
            ):
                intersection_map[ii].add(jj)
                intersection_map[jj].add(ii)
    return intersection_map


def one_vs_all_differences(geoms: list, intersection_map: dict):
    differences = []
    for ii, poly in enumerate(geoms):
        all_intersectors = unary_union([geoms[jj] for jj in intersection_map[ii]])
        differences.append(poly.difference(all_intersectors))
    return differences


def pairwise_partition_polygons(gdf: gpd.GeoDataFrame):
    total_n_intersections = 1

    disjoint_gdfs = []
    while total_n_intersections > 0:
        gdf = gdf.reset_index(drop=True)
        gdf["intersectors"] = find_intersecting_polys(gdf.geometry)
        gdf["n_intersections"] = gdf.intersectors.apply(len)

        # remove disjoint polys
        is_disjoint = gdf.n_intersections == 0
        disjoint_gdf = gdf[is_disjoint]
        if len(disjoint_gdf) > 0:
            disjoint_gdfs.append(disjoint_gdf)

        gdf = gdf[~is_disjoint]

        total_n_intersections = gdf.n_intersections.sum()
        split_gdfs = [gpd.GeoDataFrame({"geometry": []})]
        init_gdf_len = len(gdf)
        while len(gdf.query("n_intersections > 0")) > 1:

            # get one row

            seed_idx = np.random.choice(gdf.query("n_intersections > 0").index)
            poly_A = gdf.loc[seed_idx]

            # get one intersecting poly
            choices = gdf.index.intersection(poly_A.intersectors)
            if len(choices) == 0:
                break

            intersector_idx = np.random.choice(choices)
            poly_B = gdf.loc[intersector_idx]

            # split into intersection and symmetric difference
            intersection = poly_A.geometry.intersection(poly_B.geometry)
            symmetric_difference = poly_A.geometry.symmetric_difference(poly_B.geometry)

            # recombine into a gdf
            split_gdf = gpd.GeoDataFrame({"geometry": [intersection, symmetric_difference]}).explode(index_parts=False)
            split_gdf = split_gdf[split_gdf.geom_type == "Polygon"]
            split_gdf = split_gdf[~split_gdf.is_empty].reset_index(drop=True)

            split_gdfs.append(split_gdf)

            # remove used polys from gdf
            gdf = gdf.loc[~gdf.index.isin([seed_idx, intersector_idx])]
            total_n_intersections = gdf.n_intersections.sum()

            if len(gdf) == init_gdf_len:
                break

        gdf = gpd.GeoDataFrame(pd.concat(split_gdfs + [gdf]))
        gdf = gdf[gdf.geom_type == "Polygon"]
        gdf = gdf[~gdf.is_empty]
        gdf = gdf.reset_index(drop=True)

    disjoint_gdfs.append(gdf)

    gdf = gpd.GeoDataFrame(pd.concat(disjoint_gdfs)).reset_index(drop=True)
    return gdf[["geometry"]]


def find_clusters(adjacency_list):
    visited = set()
    clusters = []

    def dfs(node, cluster):
        visited.add(node)
        cluster.append(node)
        for adj_node in adjacency_list[node]:
            if adj_node not in visited:
                dfs(adj_node, cluster)

    for node in adjacency_list:
        if node not in visited:
            cluster = []
            dfs(node, cluster)
            clusters.append(cluster)

    return clusters
