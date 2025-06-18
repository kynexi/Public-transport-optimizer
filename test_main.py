import pytest
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import Point
import main

def test_coverage():
    routes = main.final_routes_gdf
    grid = main.grid_gdf
    admin = main.admin_boundary_gdf
    # Sample weighted points
    weights = grid['population'] / grid['population'].sum()
    sample = grid.sample(n=3000, weights=weights, random_state=42)
    pts = sample.geometry.centroid.to_crs(3857)
    union = routes.to_crs(3857).unary_union
    dists = pts.distance(union)
    covered = (dists <= 400).mean()
    assert covered >= 0.95, f"Coverage only {covered:.2%}, needs ≥95%"


def test_boulevard():
    routes = main.final_routes_gdf
    # get Stefan boulevard
    tags = {"name": "Ștefan cel Mare", "highway": True}
    blv = ox.geometries_from_place(main.PLACE_QUERY, tags=tags)
    blv_line = blv.geometry.unary_union.to_crs(3857)
    overlap = routes.to_crs(3857).intersection(blv_line)
    covered = overlap.length.sum() / blv_line.length
    assert covered >= 0.95, f"Boulevard coverage {covered:.0%}, needs ≥95%"


def test_delay_improvement():
    stops = main.final_stops_gdf
    routes = main.final_routes_gdf
    # build stop graph
    G = nx.Graph()
    speed = 18*1000/3600
    wait = 7.5*60
    for r, group in stops.groupby('route_name'):
        ordst = group.sort_values('stop_seq')
        for a, b in zip(ordst.geometry, ordst.geometry[1:]):
            t = Point(a).distance(Point(b)) / speed + wait
            G.add_edge(a, b, weight=t)
    idx = list(stops.geometry)
    trips = np.random.choice(idx, size=(500,2))
    times=[]
    for s,t in trips:
        try:
            times.append(nx.shortest_path_length(G, s, t, weight='weight'))
        except:
            pass
    new_avg = np.mean(times)
    old = 48*60
    assert new_avg <= old*0.8, f"Improvement only {(1-new_avg/old):.0%}, needs ≥20%"


def test_load_balance():
    stops = main.final_stops_gdf
    grid = main.grid_gdf
    capacity = 80
    demand = {}
    for _,cell in grid.iterrows():
        sid = stops.distance(cell.geometry.centroid).idxmin()
        demand[sid] = demand.get(sid,0)+cell.population
    loads = stops.copy()
    loads['demand'] = loads.index.map(lambda i: demand.get(i,0))
    peak = loads.groupby('route_name')['demand'].max()
    too_high = peak[peak > 1.2*capacity]
    assert too_high.empty, f"Overcrowded: {list(too_high.index)}"
