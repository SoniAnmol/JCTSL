import osmnx as ox
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.prepared import prep
import shapely.wkt
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely import wkt
from tqdm import tqdm
from shapely.ops import nearest_points
from credentials.credentials import *


# Method to create a grid
def grid_bounds(geom, delta):
    minx, miny, maxx, maxy = geom.bounds
    gx, gy = np.arange(start=minx, stop=maxx, step=delta), np.arange(start=miny, stop=maxy, step=delta)
    grid = []
    for i in range(len(gx) - 1):
        for j in range(len(gy) - 1):
            poly_ij = Polygon([[gx[i], gy[j]], [gx[i], gy[j + 1]], [gx[i + 1], gy[j + 1]], [gx[i + 1], gy[j]]])
            grid.append(poly_ij)
    return grid


# Method to grid a shape file
def partition(geom, delta):
    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom, delta)))
    return grid


def reset_crs(gdf, crs='EPSG:4326'):
    """Resets CRS EPSG:4326 for GeoPandas DataFrame"""
    gdf = gdf.to_crs(crs)
    return gdf


def set_local_crs(gdf, crs='EPSG:7774'):
    """Sets local CRS EPSG:7774 for GeoPandas DataFrame"""
    gdf = gdf.to_crs(crs)
    return gdf


def get_geo_dataframe(geo_series):
    """Returns a geo dataframe from geo series with crs 'EPSG:4326'"""
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geo_series), crs='EPSG:7774')
    gdf = gdf.to_crs('EPSG:4326')
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    return gdf


def polygons_from_geo_series(geo_series):
    """"Returns a GeoPandas DataFrame with Shapely Polygons from GeoSeries"""

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def xy_list_from_string(s):
        # 'x y x y ...' -> [[x, y], [x, y], ...]
        return list(chunks([float(i.strip(',')) for i in s.split()], 2))

    def poly(s):
        """ returns shapely polygon from point list"""
        ps = xy_list_from_string(s)
        return Polygon([[p[0], p[1]] for p in ps])

    polygons = [poly(r) for r in geo_series]

    return polygons


def get_nearest_nodes(gdf, graph):
    """Returns the nearest node for each origin and destination"""
    gdf['node'] = ''
    for index, row in tqdm(gdf.iterrows(), total=len(gdf)):
        gdf.loc[index, 'node'] = ox.distance.nearest_nodes(graph, X=row.geometry.centroid.x, Y=row.geometry.centroid.y)
    return gdf


def get_walking_route(trips):
    """"Find the shortest path between source and destination node on street network"""
    trips['route_length'] = ''
    for index, poi in tqdm(trips.iterrows(), total=len(trips)):
        # find the shortest path between nodes
        if poi.destination:
            route = ox.shortest_path(graph, poi.node, poi.destination, weight="travel_time")
            # route length in meters
            edge_lengths = ox.utils_graph.get_route_edge_attributes(graph, route, "length")
            trips.loc[index, 'route_length'] = round(sum(edge_lengths))
        else:
            pass
    return trips


def nearest_sap(origins, destinations, network='JCTSL'):
    """Returns a dataframe with the nearest SAP"""
    destinations.rename(columns={'name': 'sap_name'}, inplace=True)
    destinations = destinations[destinations.network == network]
    pts = destinations[destinations.network == network].geometry.unary_union
    origins.reset_index(inplace=True, drop=True)

    def near(point, pts):
        # find the nearest point and return the corresponding Place value
        nearest = destinations.geometry == nearest_points(point, pts)[1]
        return destinations.loc[nearest, ['geometry', 'node', 'sap_name', 'network', 'routes']]

    origins['nearest_sap'] = ''
    origins['destination'] = ''
    origins['sap_name'] = ''
    origins['network'] = ''
    origins['routes'] = ''

    for index, row in tqdm(origins.iterrows(), total=len(origins)):
        # Get the nearest point and its linked node
        nearest_sap = near(row.geometry.centroid, pts)
        # select the nearest point if two points are selected
        if len(nearest_sap) > 1:
            nearest_sap = nearest_sap.loc[nearest_sap.index[0], :]
        elif len(nearest_sap) == 1:
            origins.iloc[index, 6] = nearest_sap.geometry.item()
            origins.iloc[index, 7] = nearest_sap.node.item()
            origins.iloc[index, 8] = nearest_sap.sap_name.item()
            origins.iloc[index, 9] = nearest_sap.network.item()
            origins.iloc[index, 10] = nearest_sap.routes.item()

    return origins
