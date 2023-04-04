from typing import List, Union

import geopandas as gpd
import shapely.geometry as sg
from shapely import Geometry


class Distance(object):
    def __init__(self, d, unit):
        setattr(self, unit, d)

    @property
    def inches(self):
        return self._inches

    @inches.setter
    def inches(self, inches):
        self._inches = inches
        self._mm = 25.4 * inches

    @property
    def mm(self):
        return self._mm

    @mm.setter
    def mm(self, d):
        self._mm = d
        self._inches = d / 25.4


class Paper(object):
    def __init__(
        self,
        size: str = "11x14 inches",
    ):
        standard_sizes = {
            "letter": "8.5x11 inches",
            "A3": "11.7x16.5 inches",
            "A4": "8.3x11.7 inches",
            "A2": "17.01x23.42 inches",  # 432x594mm
        }

        std_size = standard_sizes.get(size, None)
        if std_size is not None:
            size = std_size
        _x, _y, _units = self.parse_size_string(size)

        self.x = Distance(_x, _units)
        self.y = Distance(_y, _units)

    @property
    def page_format_mm(self):
        return f"{self.x.mm}mmx{self.y.mm}mm"

    @staticmethod
    def parse_size_string(size_string):
        size, units = size_string.split(" ")
        x, y = size.split("x")
        return float(x), float(y), units

    def get_drawbox(self, border: float = 0, xborder=None, yborder=None):  # mm

        if xborder is None:
            xborder = border
        if yborder is None:
            yborder = border

        return sg.box(xborder, yborder, self.x.mm - xborder, self.y.mm - yborder)


def merge_LineStrings(mls_list):
    if isinstance(mls_list, Geometry):
        mls_list = mls_list.geoms
    elif isinstance(mls_list, gpd.GeoDataFrame):
        mls_list = mls_list.geometry
    elif isinstance(mls_list, gpd.GeoSeries):
        mls_list = mls_list.to_list()
    merged_mls = []
    for mls in mls_list:
        if mls.geom_type == "MultiLineString":
            merged_mls += list(mls.geoms)
        elif mls.geom_type == "LineString":
            merged_mls.append(mls)
    return sg.MultiLineString(merged_mls)


def merge_Polygons(mp_list):
    if isinstance(mp_list, Geometry):
        mp_list = mp_list.geoms
    elif isinstance(mp_list, gpd.GeoDataFrame):
        mp_list = mp_list.geometry
    elif isinstance(mp_list, gpd.GeoSeries):
        mp_list = mp_list.to_list()

    merged_mps = []
    for mp in mp_list:
        if mp.geom_type == "MultiPolygon":
            merged_mps += list(mp.geoms)
        elif mp.geom_type == "Polygon":
            merged_mps.append(mp)
        elif isinstance(mp, list):
            merged_mps += list(mp)

    return sg.MultiPolygon(merged_mps)


def flatten_geom_collection(
    geoms: Union[gpd.GeoDataFrame, gpd.GeoSeries, List[Geometry]],
    as_gdf: bool = True,
) -> Union[gpd.GeoDataFrame, List[Geometry]]:
    if isinstance(geoms, Geometry):
        if hasattr(geoms, "geoms"):
            geoms = list(geoms.geoms)
        else:  # assume Polygon or LineString
            geoms = [geoms]
    elif isinstance(geoms, gpd.GeoDataFrame):
        geoms = geoms.geometry.to_list()
    elif isinstance(geoms, gpd.GeoSeries):
        geoms = geoms.to_list()

    merged_geoms: List[Geometry] = []
    for geom in geoms:
        if isinstance(geom, list):
            merged_geoms.extend(geom)
        elif hasattr(geom, "geoms"):
            merged_geoms.extend(list(geom.geoms))
        else:
            merged_geoms.append(geom)

    if as_gdf:
        return gpd.GeoDataFrame(geometry=merged_geoms)
    else:
        return merged_geoms
