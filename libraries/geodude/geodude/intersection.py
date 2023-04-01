import itertools
from collections import deque
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm


def polys_to_gdf(polygons: Union[List, MultiPolygon]):
    try:
        if polygons.geom_type == "MultiPolygon":
            polygons = polygons.geoms
    except AttributeError:
        pass

    # Create a GeoDataFrame from the input polygons with an additional column to store the index
    return gpd.GeoDataFrame({"geometry": polygons})


def find_intersecting_polys(geoms: List[Polygon]) -> Dict[int, set]:
    intersection_map = {ii: set() for ii in range(len(geoms))}
    for ii, jj in itertools.combinations(range(len(geoms)), 2):
        if geoms[ii].buffer(-1e-6).intersects(geoms[jj].buffer(-1e-6)):
            intersection_map[ii].add(jj)
            intersection_map[jj].add(ii)
    return intersection_map


def find_intersectiong_polys_vectorized(geoms: List[Polygon]) -> Dict[int, set]:
    intersection_map = {ii: set() for ii in range(len(geoms))}
    gs = gpd.GeoSeries(geoms).buffer(-1e-6)
    for ii in range(len(geoms)):
        _gs = gs.copy()
        current = _gs.pop(ii)
        intersection_map[ii] = set(_gs.loc[_gs.intersects(current)].index)
    return intersection_map


def find_touching_polys(geoms: List[Polygon]) -> Dict[int, set]:
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


def find_touching_polys_vectorized(geoms: List[Polygon]) -> Dict[int, set]:
    intersection_map = {ii: set() for ii in range(len(geoms))}
    gs_dilated = gpd.GeoSeries(geoms).buffer(1e-6)
    gs_eroded = gpd.GeoSeries(geoms).buffer(-1e-6)
    for ii in range(len(geoms)):
        _gs_dilated = gs_dilated.copy()
        _gs_eroded = gs_eroded.copy()
        current_dilated = _gs_dilated.pop(ii)
        current_eroded = _gs_eroded.pop(ii)
        intersects_dilated = _gs_dilated.intersects(current_dilated)
        intersects_eroded = _gs_eroded.intersects(current_eroded)
        touches_idx = intersects_dilated & ~intersects_eroded
        intersection_map[ii] = set(_gs_dilated.loc[touches_idx].index)
    return intersection_map


def find_contained_polys(geoms: List[Polygon]) -> Dict[int, set]:
    contains_map = {ii: set() for ii in range(len(geoms))}
    for ii, jj in itertools.product(range(len(geoms)), range(len(geoms))):
        if ii != jj:
            if geoms[ii].buffer(1e-6).contains(geoms[jj].buffer(1e-7)):
                contains_map[ii].add(jj)
    return contains_map


def one_vs_all_differences(geoms: List[Polygon], intersection_map: Dict[int, set]) -> List[Polygon]:
    differences = []
    for ii, poly in enumerate(geoms):
        all_intersectors = unary_union([geoms[jj] for jj in intersection_map[ii]])
        differences.append(poly.difference(all_intersectors))
    return differences


def pairwise_partition_polygons(
    gdf: gpd.GeoDataFrame,
    verbose: bool = False,
    min_area: float = None,
) -> Tuple[gpd.GeoDataFrame, List[int]]:
    total_n_intersections = 1

    disjoint_gdfs = []
    n_intersections_log = []
    while total_n_intersections > 0:
        gdf = gdf.reset_index(drop=True)
        gdf["intersectors"] = find_intersectiong_polys_vectorized(gdf.geometry)
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
            n_intersections_log.append(total_n_intersections)
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
            if verbose:
                print(total_n_intersections)
            if len(gdf) == init_gdf_len:
                break

        gdf = gpd.GeoDataFrame(pd.concat(split_gdfs + [gdf]))
        gdf = gdf[gdf.geom_type == "Polygon"]
        gdf = gdf[~gdf.is_empty]
        if min_area is not None:
            gdf = gdf[gdf.area > min_area]
        gdf = gdf.reset_index(drop=True)

    disjoint_gdfs.append(gdf)

    gdf = gpd.GeoDataFrame(pd.concat(disjoint_gdfs)).reset_index(drop=True)
    return gdf[["geometry"]], n_intersections_log


def find_parent_polygons(
    disjoint: gpd.GeoDataFrame,
    original: gpd.GeoDataFrame,
    buffer_distance: float = -1e-6,
    min_norm_area: float = 0.5,  # theoretically this should be 1.0
) -> gpd.GeoDataFrame:
    disjoint["adjacent_polys"] = find_touching_polys_vectorized(disjoint.geometry)
    disjoint.loc[:, "intersecting_original_polygons"] = [set() for _ in range(len(disjoint))]
    disjoint["order"] = [tuple() for _ in range(len(disjoint))]
    slightly_buffered_gdf = original.buffer(buffer_distance)

    # Iterate over each disjoint polygon
    for ii, row in disjoint.iterrows():
        poly = row["geometry"]

        # Find the original polygons that intersect with the new polygon
        candidate_parents = set(slightly_buffered_gdf[slightly_buffered_gdf.intersects(poly)].index)
        parents = set()

        for jj in candidate_parents:
            area = poly.intersection(original.loc[jj].geometry).area / poly.area
            if area > min_norm_area:
                parents.add(jj)

        disjoint.loc[ii, "intersecting_original_polygons"].update(parents)

    disjoint["n_parents"] = disjoint.intersecting_original_polygons.apply(len)
    idx = disjoint["n_parents"] == 1
    disjoint.loc[idx, "order"] = disjoint.loc[idx, "intersecting_original_polygons"].apply(lambda x: tuple(x))
    disjoint = disjoint.sort_values("n_parents", ascending=False)

    return disjoint


def assign_psuedoperiodic_order_to_adjacent_clusters(disjoint):
    idx = disjoint["n_parents"] == 1
    is_an_overlap = disjoint.loc[~idx]
    overlap_idx = set(is_an_overlap.index)
    valid_adjacency_list = {
        k: v.intersection(overlap_idx) for k, v in is_an_overlap["adjacent_polys"].to_dict().items()
    }
    clusters = find_clusters(valid_adjacency_list)

    all_parents = set().union(*disjoint["intersecting_original_polygons"].to_list())
    order = deque(all_parents)
    for cluster in clusters:
        valid_parents = set().union(*disjoint.loc[cluster]["intersecting_original_polygons"].to_list())
        local_order = [x for x in order if x in valid_parents]
        for ii in cluster:
            disjoint.at[ii, "order"] = tuple(
                parent for parent in local_order if parent in disjoint.loc[ii, "intersecting_original_polygons"]
            )
        order.rotate()
    disjoint["parent"] = disjoint["order"].apply(lambda x: x[0])
    return disjoint


def assign_random_order_to_adjacent_clusters(disjoint):
    idx = disjoint["n_parents"] == 1
    is_an_overlap = disjoint.loc[~idx]
    overlap_idx = set(is_an_overlap.index)
    valid_adjacency_list = {
        k: v.intersection(overlap_idx) for k, v in is_an_overlap["adjacent_polys"].to_dict().items()
    }
    clusters = find_clusters(valid_adjacency_list)

    for ii, cluster in enumerate(clusters):
        valid_parents = set().union(*disjoint.loc[cluster]["intersecting_original_polygons"].to_list())
        local_order = tuple(np.random.permutation(list(valid_parents)))
        for ii in cluster:
            disjoint.at[ii, "order"] = tuple(
                parent for parent in local_order if parent in disjoint.loc[ii, "intersecting_original_polygons"]
            )

    def get_first(x):
        try:
            return x[0]
        except IndexError:
            return -1

    disjoint["parent"] = disjoint["order"].apply(get_first)
    return disjoint


def merge_disjoint_polys(disjoint: gpd.GeoDataFrame):
    new_polys = []
    for parent, sub_gdf in disjoint.groupby("parent"):
        merged = unary_union(sub_gdf.geometry)

        if isinstance(merged, Polygon):
            new_polys.append(merged)
        else:

            new_polys.extend(merged.geoms)
    return gpd.GeoDataFrame(geometry=new_polys)


def find_clusters(adjacency_list: Dict[int, set]) -> List[List[int]]:
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


def chunked_pairwise_partition_polygons(gdf: gpd.GeoDataFrame, chunk_size: int = 100, **kwargs):
    total_n_intersections = 1
    iteration_num = 0
    disjoint_gdfs = []
    while total_n_intersections > 0:
        iteration_num += 1
        gdf = gdf.reset_index(drop=True)
        print(f"Iteration {iteration_num}")
        print(f"Finding intersections in {len(gdf)} polygons")
        gdf["intersectors"] = find_intersectiong_polys_vectorized(gdf.geometry)
        gdf["n_intersections"] = gdf.intersectors.apply(len)
        # remove disjoint polys
        is_disjoint = gdf.n_intersections == 0
        disjoint_gdf = gdf[is_disjoint]
        if len(disjoint_gdf) > 0:
            disjoint_gdfs.append(disjoint_gdf)

        gdf = gdf[~is_disjoint]

        total_n_intersections = gdf.n_intersections.sum()
        print(f"{total_n_intersections} intersections remaining")
        chunks = []
        used_indices = set()
        print("Chunking polygons")
        while len(used_indices) < len(gdf):
            # pick random row to start with
            global_indices_to_pick_from = set(gdf.index) - used_indices
            if len(global_indices_to_pick_from) == 0:
                break
            current_chunk = gdf.loc[list(global_indices_to_pick_from)].sample(1)
            used_indices.update(current_chunk.index.to_list())
            while len(current_chunk) < chunk_size:

                global_indices_to_pick_from = set(gdf.index) - used_indices
                current_chunk_intersectors = set().union(*current_chunk.intersectors)
                current_indices_to_pick_from = current_chunk_intersectors - used_indices

                n_additions = min(len(current_indices_to_pick_from), chunk_size - len(current_chunk))
                if n_additions > 0:
                    to_add_idx = np.random.choice(
                        list(current_indices_to_pick_from),
                        n_additions,
                        replace=False,
                    )
                    current_chunk = pd.concat([current_chunk, gdf.loc[to_add_idx]])
                elif n_additions == 0:
                    if len(global_indices_to_pick_from) == 0:
                        used_indices.update(current_chunk.index.to_list())
                        break
                    current_chunk = pd.concat([current_chunk, gdf.loc[list(global_indices_to_pick_from)].sample(1)])

                used_indices.update(current_chunk.index.to_list())
                global_indices_to_pick_from = set(gdf.index) - used_indices

            chunks.append(current_chunk)
            # print(f'Chunk {len(chunks)}: {len(current_chunk)} polygons, {len(global_indices_to_pick_from)} polygons remaining')

        if len(chunks) == 0:
            break

        print(f"Partitioning {len(chunks)} chunks")
        for ii, chunk in tqdm(enumerate(chunks)):
            # current_chunk_intersections = chunk.n_intersections.sum()
            # print(
            #     f"Partitioning chunk {ii+1}/{len(chunks)} ({len(chunk)} polygons, {current_chunk_intersections} intersections)"
            # )
            chunk, _ = pairwise_partition_polygons(chunk, **kwargs)
            chunks[ii] = chunk

        # recombine
        gdf = gpd.GeoDataFrame(pd.concat(chunks))

        # scramble
        gdf = gdf.sample(frac=1, replace=False).reset_index(drop=True)
        print(f"Iteration {iteration_num} complete, {len(gdf)} polygons remaining")
    disjoint_gdfs.append(gdf)

    gdf = gpd.GeoDataFrame(pd.concat(disjoint_gdfs)).reset_index(drop=True)

    return gdf
