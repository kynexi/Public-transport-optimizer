import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import AntPath, HeatMap
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
import pyproj
import random
import numpy as np

print("Starting the 'Realistic' Chișinău Bus Network Demonstrator Script...")

# --- 0. Configuration ---
PLACE_QUERY = "Chișinău, Moldova"
NETWORK_TYPE = 'drive' # For roads buses can use
PROJECTED_CRS = "EPSG:32635"  # WGS 84 / UTM zone 35N
WGS84_CRS = "EPSG:4326"
STOP_SPACING_METERS = 400
MIN_ROUTE_LENGTH_METERS = 2000 
MAX_ROUTES_TO_GENERATE = 25    # Cap for visualization clarity
NUM_POP_POINTS_PER_SECTOR_BASE = 10 # Base number of population points per sector, will be weighted

# Pyproj transformers
TRANSFORMER_TO_PROJ = pyproj.Transformer.from_crs(WGS84_CRS, PROJECTED_CRS, always_xy=True).transform
TRANSFORMER_TO_WGS84 = pyproj.Transformer.from_crs(PROJECTED_CRS, WGS84_CRS, always_xy=True).transform

# --- 1. Data Fetching & Preparation Functions ---
def fetch_base_data(place_query, network_type):
    print(f"Fetching base data for {place_query}...")
    admin_boundary_gdf = ox.geocode_to_gdf(place_query) # Entire Chișinău municipality
    chisinau_polygon = admin_boundary_gdf.unary_union
    
    G_roads = ox.graph_from_polygon(chisinau_polygon, network_type=network_type, retain_all=False, simplify=True)
    G_roads_proj = ox.project_graph(G_roads, to_crs=PROJECTED_CRS)
    print(f"Road network: {len(G_roads.nodes)} nodes, {len(G_roads.edges)} edges.")

    tags = { # Comprehensive POI tags
        'amenity': ['hospital', 'university', 'school', 'marketplace', 'bus_station', 'police', 'fire_station', 'theatre', 'library', 'bank', 'clinic', 'government', 'post_office', 'community_centre', 'social_facility', 'kindergarten'],
        'shop': ['mall', 'supermarket', 'department_store', 'retail'],
        'office': True,
        'tourism': ['hotel', 'museum', 'attraction', 'artwork'],
        'public_transport': ['station', 'platform', 'stop_position'], # Railway station, major bus terminals
        'sport': ['stadium', 'sports_centre', 'pitch', 'fitness_centre'],
        'leisure': ['park', 'playground', 'garden', 'cinema'],
        'building': ['commercial', 'industrial', 'office', 'retail', 'public', 'train_station'],
        'landuse': ['commercial', 'industrial', 'retail', 'institutional']
    }
    pois_gdf = ox.features_from_polygon(chisinau_polygon, tags=tags)
    # Use centroids for polygon POIs, ensure they are points
    pois_gdf['geometry'] = pois_gdf.apply(lambda row: row.geometry.centroid if row.geometry.type != 'Point' else row.geometry, axis=1)
    pois_gdf = pois_gdf[pois_gdf.geometry.type == 'Point']
    print(f"Fetched {len(pois_gdf)} POIs.")
    return admin_boundary_gdf, G_roads, G_roads_proj, pois_gdf

def define_chisinau_sectors(city_centroid_wgs84):
    """
    Defines improved polygons for Chișinău's main sectors, including Telecentru.
    Coordinates are WGS84 (lon, lat).
    """
    print("Defining improved sector polygons...")
    cx, cy = city_centroid_wgs84.x, city_centroid_wgs84.y

    # Mircea cel Bătrân Blvd. endpoint (Ciocana extreme N/NE)
    mircea_end = (28.890937, 47.055966)

    sectors_data = {
        "Centru": {
            "polygon": Polygon([
                (cx-0.010, cy+0.010), (cx+0.010, cy+0.010),
                (cx+0.010, cy-0.010), (cx-0.010, cy-0.010)
            ]), "pop_percentage": 0.13, "color": "red"
        },
        "Buiucani": {
            "polygon": Polygon([
                (cx-0.018, cy+0.018), (cx-0.050, cy+0.050),
                (cx-0.090, cy+0.030), (cx-0.100, cy-0.010),
                (cx-0.080, cy-0.030),
                (cx-0.040, cy-0.018),
                (cx-0.012, cy-0.012)
            ]), "pop_percentage": 0.16, "color": "blue"
        },
        "Râșcani": {
            "polygon": Polygon([
                (cx+0.010, cy+0.010), 
                (cx-0.005, cy+0.045),
                (cx+0.015, cy+0.060),
                (cx+0.045, cy+0.055),
                (cx+0.060, cy+0.040),
                (cx+0.050, cy+0.025),
                (cx+0.030, cy+0.012),
                (cx+0.010, cy-0.010)
            ]), "pop_percentage": 0.20, "color": "purple"
        },
            "Ciocana": {
            "polygon": Polygon([
                (28.885130401685444, 47.07622648248136),
                (28.886146984970452, 47.058744764766544),
                (28.88035940675519, 47.048252346202645),
                (28.877237445573144, 47.03019164164252),
                (28.86768558915975, 47.0240366308399),
                (28.879725767625303, 47.01002732974857),
                (28.90238284912533, 46.99623745556667),
                (28.9106720798126, 46.994236840823476),
                (28.90286697676089, 47.021337808435305),
                (28.90027302696413, 47.07632776807773),
                (28.893201544157364, 47.07620531422853),
                (28.885130401685444, 47.07622648248136)
            ]), "pop_percentage": 0.16, "color": "orange"
            },
        "Botanica": {
            "polygon": Polygon([
                (cx+0.010, cy-0.010), (cx+0.040, cy-0.018),
                (cx+0.090, cy-0.030), (cx+0.070, cy-0.080),
                (cx-0.010, cy-0.070), (cx-0.014, cy-0.012)
            ]), "pop_percentage": 0.25, "color": "green"
        },
        "Telecentru": {
            "polygon": Polygon([
                (cx-0.010, cy-0.010), (cx-0.014, cy-0.012),
                (cx-0.040, cy-0.018), (cx-0.085, cy-0.010),
                (cx-0.090, cy-0.050), (cx-0.050, cy-0.080), (cx-0.010, cy-0.070)
            ]), "pop_percentage": 0.10, "color": "brown"
        }
    }

    sector_list = []
    for name, data in sectors_data.items():
        sector_list.append({
            'name': name,
            'geometry': data["polygon"],
            'pop_percentage': data["pop_percentage"],
            'color': data["color"]
        })
    sectors_gdf = gpd.GeoDataFrame(sector_list, crs=WGS84_CRS)
    print(f"Defined {len(sectors_gdf)} improved sector polygons (including Telecentru).")
    return sectors_gdf

def generate_population_points(sectors_gdf, base_points_per_sector):
    """Generates weighted population points within each sector."""
    print("Generating mock population points within sectors...")
    all_pop_points = []
    for _, sector_row in sectors_gdf.iterrows():
        num_points = int(base_points_per_sector * (sector_row['pop_percentage'] / (1.0/len(sectors_gdf))) * 5) # Scale up total points
        
        minx, miny, maxx, maxy = sector_row.geometry.bounds
        points_generated = 0
        attempts = 0
        sector_pop_points = []

        while points_generated < num_points and attempts < num_points * 10:
            attempts += 1
            # Generate points somewhat clustered towards the centroid of the sector
            # by using a normal distribution around the centroid, clipped by bounds
            mean_x, mean_y = sector_row.geometry.centroid.x, sector_row.geometry.centroid.y
            std_dev_x = (maxx - minx) / 4 # Spread points within the sector
            std_dev_y = (maxy - miny) / 4
            
            rand_lon = np.random.normal(mean_x, std_dev_x)
            rand_lat = np.random.normal(mean_y, std_dev_y)
            point = Point(rand_lon, rand_lat)
            
            if sector_row.geometry.contains(point):
                sector_pop_points.append({'geometry': point, 'sector': sector_row['name'], 'type': 'population_origin'})
                points_generated += 1
        all_pop_points.extend(sector_pop_points)
        
    pop_points_gdf = gpd.GeoDataFrame(all_pop_points, crs=WGS84_CRS)
    print(f"Generated {len(pop_points_gdf)} mock population points across sectors.")
    return pop_points_gdf

def identify_key_poi_hubs(pois_gdf, sectors_gdf):
    """Identifies major POI hubs: CBD, main railway, bus station, airport, university, hospital."""
    hubs = {'poi_clusters': {}}
    # CBD Proxy
    centru_sector = sectors_gdf[sectors_gdf['name'] == 'Centru']
    if not centru_sector.empty:
        hubs['cbd_proxy'] = Point(28.8328014700474, 47.02490252911852)
        print(f"CBD proxy identified at: {hubs['cbd_proxy']}")

    # Main Railway Station
    

    # Central Bus Station
    bus_station = pois_gdf[pois_gdf['name'].str.contains("Gara Auto Centru", case=False, na=False)]
    if not bus_station.empty:
        hubs['bus_station'] = bus_station.geometry.iloc[0]
        print(f"Central bus station identified at: {hubs['bus_station']}")

    # Airport
    airport = pois_gdf[pois_gdf['name'].str.contains("Aeroportul Internațional", case=False, na=False)]
    if not airport.empty:
        hubs['airport'] = airport.geometry.iloc[0]
        print(f"Airport identified at: {hubs['airport']}")

    # Main University
    university = pois_gdf[pois_gdf['name'].str.contains("Universitatea de Stat", case=False, na=False)]
    if not university.empty:
        hubs['university'] = university.geometry.iloc[0]
        print(f"Main university identified at: {hubs['university']}")

    # Main Hospital
    hospital = pois_gdf[pois_gdf['name'].str.contains("Spitalul Clinic Republican", case=False, na=False)]
    if not hospital.empty:
        hubs['hospital'] = hospital.geometry.iloc[0]
        print(f"Main hospital identified at: {hubs['hospital']}")

    return hubs
# --- 2. Route Generation Logic (More Realistic Heuristics) ---
def generate_route_and_stops_detailed(G_roads, origin_geom, dest_geom, route_name, min_route_len_m):
    """Generates a single route and its stops with detailed error handling."""
    # Access global transformers defined outside
    global TRANSFORMER_TO_PROJ, TRANSFORMER_TO_WGS84, STOP_SPACING_METERS
    try:
        orig_node = ox.nearest_nodes(G_roads, X=origin_geom.x, Y=origin_geom.y)
        dest_node = ox.nearest_nodes(G_roads, X=dest_geom.x, Y=dest_geom.y)

        if orig_node == dest_node: return None, None
        
        route_nodes = nx.shortest_path(G_roads, source=orig_node, target=dest_node, weight='length')
        if len(route_nodes) < 2 : return None, None # Path must have at least 2 nodes

        route_line = LineString([(G_roads.nodes[node]['x'], G_roads.nodes[node]['y']) for node in route_nodes])
        route_line_proj = transform(TRANSFORMER_TO_PROJ, route_line)
        route_length_m = route_line_proj.length

        if route_length_m < min_route_len_m: return None, None
        
        route_feature = {'geometry': route_line, 'name': route_name, 'length_m': route_length_m}
        stops = []
        # Add first stop
        stop_point_proj_start = route_line_proj.interpolate(0)
        stops.append({'geometry': transform(TRANSFORMER_TO_WGS84, stop_point_proj_start), 'route_name': route_name, 'stop_seq': 1, 'type': 'start'})
        
        current_dist = STOP_SPACING_METERS
        stop_counter = 2
        while current_dist < route_length_m:
            stop_point_proj = route_line_proj.interpolate(current_dist)
            stops.append({'geometry': transform(TRANSFORMER_TO_WGS84, stop_point_proj), 'route_name': route_name, 'stop_seq': stop_counter, 'type': 'intermediate'})
            current_dist += STOP_SPACING_METERS
            stop_counter += 1
            
        # Add last stop if not too close to the previous one
        last_inter_stop_proj = transform(TRANSFORMER_TO_PROJ, stops[-1]['geometry']) if len(stops) > 1 else stop_point_proj_start
        stop_point_proj_end = route_line_proj.interpolate(route_length_m)
        if last_inter_stop_proj.distance(stop_point_proj_end) > STOP_SPACING_METERS / 3:
            stops.append({'geometry': transform(TRANSFORMER_TO_WGS84, stop_point_proj_end), 'route_name': route_name, 'stop_seq': stop_counter if stops[-1]['type'] != 'end' else stops[-1]['stop_seq']+1 , 'type': 'end'})
        elif stops[-1]['type'] != 'end' : # if last stop is not already an 'end' stop, update it
            stops[-1]['type'] = 'end'
            stops[-1]['geometry'] = transform(TRANSFORMER_TO_WGS84, stop_point_proj_end)


        # print(f"  Generated route {route_name} ({route_length_m:.0f}m), {len(stops)} stops.")
        return route_feature, stops
    except nx.NetworkXNoPath:
        # print(f"  No path for {route_name}.")
        return None, None
    except Exception as e:
        print(f"  Error for route {route_name}: {e}")
        return None, None

# --- 3. Map Visualization Functions (largely same as before, refined popups) ---
def create_base_map(center_coords, zoom=12):
    return folium.Map(location=center_coords, zoom_start=zoom, tiles="CartoDB positron")

def add_sectors_to_map(m, sectors_gdf):
    if not sectors_gdf.empty:
        layer = folium.FeatureGroup(name="Chișinău Sectors (Approx.)")
        for _, row in sectors_gdf.iterrows():
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, color=row['color']: {'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.2},
                tooltip=f"Sector: {row['name']}<br>Pop. Share: {row['pop_percentage']*100:.1f}%"
            ).add_to(layer)
        layer.add_to(m)

def add_population_points_heatmap(m, pop_points_gdf):
    if not pop_points_gdf.empty:
        # For HeatMap, data format is list of [lat, lon] or [lat, lon, weight]
        heat_data = [[point.y, point.x] for point in pop_points_gdf.geometry]
        HeatMap(heat_data, name="Population Density", radius=15, blur=10).add_to(m)

def add_pois_to_map_detailed(m, pois_gdf, key_hubs):
    # Generic POIs
    if not pois_gdf.empty:
        layer = folium.FeatureGroup(name="Points of Interest (Sample)")
        # Sample POIs to avoid clutter, prioritize named ones
        sample_pois = pois_gdf[pois_gdf['name'].notna()].sample(min(len(pois_gdf[pois_gdf['name'].notna()]), 200), random_state=1) if len(pois_gdf[pois_gdf['name'].notna()]) > 0 else pois_gdf.sample(min(len(pois_gdf), 200), random_state=1)

        for _, row in sample_pois.iterrows():
            if row.geometry.geom_type == 'Point':
                popup_text = f"<b>{row.get('name', 'N/A')}</b>"
                tags_of_interest = ['amenity', 'shop', 'office', 'tourism', 'public_transport', 'sport', 'leisure', 'building', 'landuse']
                type_info = [f"{tag.capitalize()}: {row.get(tag)}" for tag in tags_of_interest if row.get(tag) and isinstance(row.get(tag), str)]
                if type_info: popup_text += "<br>" + "<br>".join(type_info[:2]) # Show first 2 relevant tags
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x], radius=3, color='gray',
                    fill=True, fill_color='lightgray', fill_opacity=0.6,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(layer)
        layer.add_to(m)
    
    # Key Hubs
    hub_layer = folium.FeatureGroup(name="Key Hubs / Destinations")
    if 'cbd_proxy' in key_hubs:
        folium.Marker(location=[key_hubs['cbd_proxy'].y, key_hubs['cbd_proxy'].x], popup="CBD Proxy (Centru)", icon=folium.Icon(color='red', icon='building')).add_to(hub_layer)
    if 'railway_station' in key_hubs:
        folium.Marker(location=[key_hubs['railway_station'].y, key_hubs['railway_station'].x], popup="Gara Feroviară", icon=folium.Icon(color='blue', icon='train')).add_to(hub_layer)
    if 'bus_stations' in key_hubs:
        for bs_geom in key_hubs['bus_stations'][:3]: # Show up to 3
            folium.Marker(location=[bs_geom.y, bs_geom.x], popup="Bus Station", icon=folium.Icon(color='orange', icon='bus')).add_to(hub_layer)
    
    poi_cluster_colors = {'commercial':'green', 'education':'purple', 'health':'pink'}
    for cat_name, points in key_hubs.get('poi_clusters', {}).items():
        for pt_geom in points:
             folium.CircleMarker(location=[pt_geom.y, pt_geom.x], radius=5, color=poi_cluster_colors.get(cat_name, 'black'), fill=True, fill_opacity=0.8, popup=f"{cat_name.capitalize()} Cluster Area").add_to(hub_layer)
    hub_layer.add_to(m)


def add_routes_and_stops_to_map_detailed(m, routes_gdf, stops_gdf):
    # (Similar to previous, ensure popups are good)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # Tableau10
    if not routes_gdf.empty:
        layer = folium.FeatureGroup(name="New Proposed Bus Routes")
        for idx, row in routes_gdf.iterrows():
            path_coords = [(lat, lon) for lon, lat in row.geometry.coords]
            AntPath(
                locations=path_coords, dash_array=[10, 20], delay=800, weight=5,
                color=colors[idx % len(colors)], pulse_color=colors[(idx+1) % len(colors)], # Shift pulse color
                popup=f"<b>Route: {row['name']}</b><br>Length: {row['length_m']:.0f}m"
            ).add_to(layer)
        layer.add_to(m)

    if not stops_gdf.empty:
        stops_layer_fg = folium.FeatureGroup(name="New Proposed Bus Stops")
        for _, row in stops_gdf.iterrows():
            # Determine color by type
            stop_color = 'darkblue'
            if row['type'] == 'start': stop_color = 'green'
            elif row['type'] == 'end': stop_color = 'red'

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x], radius=3, color=stop_color,
                fill=True, fill_color=stop_color, fill_opacity=0.8,
                popup=f"Stop for Route: {row['route_name']}<br>Seq: {row['stop_seq']} ({row['type']})"
            ).add_to(stops_layer_fg)
        stops_layer_fg.add_to(m)


# --- 4. Main Script Logic ---
if __name__ == "__main__":
    admin_boundary_gdf, G_roads, G_roads_proj, pois_gdf = fetch_base_data(PLACE_QUERY, NETWORK_TYPE)
    
    city_center_lon, city_center_lat = 28.8328014700474, 47.02490252911852
    city_centroid_wgs84 = Point(city_center_lon, city_center_lat)
    sectors_gdf = define_chisinau_sectors(city_centroid_wgs84)
    pop_points_gdf = generate_population_points(sectors_gdf, NUM_POP_POINTS_PER_SECTOR_BASE)
    mircea_end = (28.890937, 47.055966)
    mircea_point = gpd.GeoDataFrame(
    [{'geometry': Point(mircea_end), 'sector': 'Ciocana', 'type': 'population_origin'}],
    crs=WGS84_CRS
    )
    # After generating pop_points_gdf and mircea_point
    special_demand_point = gpd.GeoDataFrame(
    [{
        'geometry': Point(28.804152431922205, 47.03920945254839),
        'sector': 'Buiucani',  
        'type': 'population_origin'
    }],
    crs=WGS84_CRS
    )
    extra_ciocana_points = [
    (28.890177330242153, 47.05002464577251),
    (28.889933143629065, 47.04113849131693),
    (28.88721388560191, 47.015501071097844),
    (28.88706071502901, 47.01460811488502)
    ]
    extra_ciocana_gdf = gpd.GeoDataFrame(
        [{
            'geometry': Point(lon, lat),
            'sector': 'Ciocana',
            'type': 'population_origin'
        } for lon, lat in extra_ciocana_points],
        crs=WGS84_CRS
    )
    pop_points_gdf = pd.concat([
    pop_points_gdf,
    mircea_point,
    special_demand_point,   
    extra_ciocana_gdf
    ], ignore_index=True)

    key_hubs = identify_key_poi_hubs(pois_gdf, sectors_gdf)

    all_new_routes = []
    all_new_stops = []
    generated_route_names = set() # To avoid duplicate route names if logic overlaps
    route_id_counter = 1

    # Strategy 1: Connect Sector Centers to CBD and to each other (Trunk-like)
    sector_centroids = {name: geom.centroid for name, geom in zip(sectors_gdf['name'], sectors_gdf['geometry'])}
    if 'cbd_proxy' in key_hubs:
        for sector_name, sector_centroid in sector_centroids.items():
            if route_id_counter > MAX_ROUTES_TO_GENERATE: break
            if sector_name != "Centru": # Don't connect Centru to itself this way
                r_name = f"Trunk_{sector_name}_to_CBD"
                if r_name not in generated_route_names:
                    route_f, stops_f = generate_route_and_stops_detailed(G_roads, sector_centroid, key_hubs['cbd_proxy'], r_name, MIN_ROUTE_LENGTH_METERS)
                    if route_f:
                        all_new_routes.append(route_f); all_new_stops.extend(stops_f); generated_route_names.add(r_name); route_id_counter +=1
    
    # Connect adjacent peripheral sectors (example pairs)
    peripheral_sectors = ["Botanica", "Buiucani", "Râșcani", "Ciocana"]
    # Example: Botanica-Rascani, Buiucani-Ciocana (more cross-town)
    adj_pairs = [("Botanica", "Râșcani"), ("Buiucani", "Ciocana"), ("Botanica", "Buiucani"), ("Râșcani", "Ciocana")]
    for s1_name, s2_name in adj_pairs:
        if route_id_counter > MAX_ROUTES_TO_GENERATE: break
        r_name = f"Trunk_{s1_name}_to_{s2_name}"
        if r_name not in generated_route_names and s1_name in sector_centroids and s2_name in sector_centroids:
            route_f, stops_f = generate_route_and_stops_detailed(G_roads, sector_centroids[s1_name], sector_centroids[s2_name], r_name, MIN_ROUTE_LENGTH_METERS * 1.5) # Longer min length
            if route_f:
                all_new_routes.append(route_f); all_new_stops.extend(stops_f); generated_route_names.add(r_name); route_id_counter +=1

    # Strategy 2: Connect Population Points to Key Hubs/POI Clusters (Connector-like)
    if not pop_points_gdf.empty:
        sampled_pop_points = pop_points_gdf.sample(min(len(pop_points_gdf), MAX_ROUTES_TO_GENERATE*2), random_state=42) # Sample to limit combinations

        for _, pop_row in sampled_pop_points.iterrows():
            if route_id_counter > MAX_ROUTES_TO_GENERATE: break
            pop_origin_geom = pop_row.geometry
            pop_sector_name = pop_row.sector

            # A) To CBD
            if 'cbd_proxy' in key_hubs:
                r_name = f"Conn_{pop_sector_name}{route_id_counter}_to_CBD"
                if r_name not in generated_route_names:
                    route_f, stops_f = generate_route_and_stops_detailed(G_roads, pop_origin_geom, key_hubs['cbd_proxy'], r_name, MIN_ROUTE_LENGTH_METERS)
                    if route_f: all_new_routes.append(route_f); all_new_stops.extend(stops_f); generated_route_names.add(r_name); route_id_counter +=1
            if route_id_counter > MAX_ROUTES_TO_GENERATE: break

            # B) To Railway Station
            if 'railway_station' in key_hubs:
                r_name = f"Conn_{pop_sector_name}{route_id_counter}_to_RailSt"
                if r_name not in generated_route_names:
                    route_f, stops_f = generate_route_and_stops_detailed(G_roads, pop_origin_geom, key_hubs['railway_station'], r_name, MIN_ROUTE_LENGTH_METERS)
                    if route_f: all_new_routes.append(route_f); all_new_stops.extend(stops_f); generated_route_names.add(r_name); route_id_counter +=1
            if route_id_counter > MAX_ROUTES_TO_GENERATE: break

            # C) To a random POI Cluster Type
            if key_hubs.get('poi_clusters'):
                chosen_cat = random.choice(list(key_hubs['poi_clusters'].keys()))
                if key_hubs['poi_clusters'][chosen_cat]:
                    chosen_poi_dest = random.choice(key_hubs['poi_clusters'][chosen_cat])
                    r_name = f"Conn_{pop_sector_name}{route_id_counter}_to_{chosen_cat[:4]}"
                    if r_name not in generated_route_names:
                        route_f, stops_f = generate_route_and_stops_detailed(G_roads, pop_origin_geom, chosen_poi_dest, r_name, MIN_ROUTE_LENGTH_METERS)
                        if route_f: all_new_routes.append(route_f); all_new_stops.extend(stops_f); generated_route_names.add(r_name); route_id_counter +=1
            if route_id_counter > MAX_ROUTES_TO_GENERATE: break

    final_routes_gdf = gpd.GeoDataFrame(all_new_routes[:MAX_ROUTES_TO_GENERATE], crs=WGS84_CRS) if all_new_routes else gpd.GeoDataFrame(columns=['geometry', 'name', 'length_m'], crs=WGS84_CRS)
    # Filter stops to only include those for the selected routes
    if all_new_stops and not final_routes_gdf.empty:
        selected_route_names = set(final_routes_gdf['name'])
        final_stops = [stop for stop in all_new_stops if stop['route_name'] in selected_route_names]
        final_stops_gdf = gpd.GeoDataFrame(final_stops, crs=WGS84_CRS) if final_stops else gpd.GeoDataFrame(columns=['geometry', 'route_name', 'stop_seq', 'type'], crs=WGS84_CRS)
    else:
        final_stops_gdf = gpd.GeoDataFrame(columns=['geometry', 'route_name', 'stop_seq', 'type'], crs=WGS84_CRS)


    print(f"Final routes generated and selected: {len(final_routes_gdf)}")
    print(f"Final stops for selected routes: {len(final_stops_gdf)}")

    # Create and save map
    map_center = [city_centroid_wgs84.y, city_centroid_wgs84.x]
    m = create_base_map(map_center, zoom=12)
    
    add_sectors_to_map(m, sectors_gdf)
    if not pop_points_gdf.empty: add_population_points_heatmap(m, pop_points_gdf)
    add_pois_to_map_detailed(m, pois_gdf, key_hubs)
    add_routes_and_stops_to_map_detailed(m, final_routes_gdf, final_stops_gdf)
    
    folium.LayerControl(collapsed=False).add_to(m)
    output_filename = "chisinau_realistic_bus_network_demo.html"
    m.save(output_filename)
    print(f"Map saved to {output_filename}. Open this file in a web browser.")
    print("Script finished.")