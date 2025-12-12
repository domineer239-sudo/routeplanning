import osmnx as ox
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
import datetime

warnings.filterwarnings("ignore")

# ==========================================
# 1. Config
# ==========================================
# Target directory for output files (Adjust this path as needed)
TARGET_DIR = r"D:\202509的课件\ai作业" 
TARGET_LOCATION = (5.4195, 100.3325) # Location near Georgetown, Penang
SEARCH_RADIUS = 1200 # Radius in meters

# ==========================================
# 2. Helpers
# ==========================================
def get_bearing(lat1, lon1, lat2, lon2):
    """Calculates the initial bearing between two GPS coordinates."""
    lat1 = math.radians(lat1); lat2 = math.radians(lat2)
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    brng = math.atan2(y, x)
    return (math.degrees(brng) + 360) % 360

def clean_attribute(val, default_val):
    """Cleans up list-based or missing OSM attributes."""
    if isinstance(val, list): return val[0] 
    if pd.isna(val) or val is None: return default_val
    return val

def categorize_road(osm_type):
    """Categorizes OSM highway types into simplified classes."""
    t = str(osm_type)
    if 'motorway' in t or 'trunk' in t: return "Highway"
    if 'primary' in t: return "Main_Artery"
    if 'secondary' in t: return "Secondary_Road"
    if 'tertiary' in t: return "Connector_Road"
    if 'residential' in t or 'living_street' in t: return "Local_Street"
    return "Small_Alley"

def generate_fake_timestamp(scenario):
    """Generates a plausible timestamp based on the traffic scenario."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    if scenario == "Morning_Peak": h = random.randint(7, 9)
    elif scenario == "Evening_Peak": h = random.randint(17, 19)
    else: h = random.choice([random.randint(10, 15), random.randint(21, 23)])
    m = random.randint(0, 59); s = random.randint(0, 59)
    return f"{date_str} {h:02d}:{m:02d}:{s:02d}"

# ==========================================
# 3. Traffic Logic
# ==========================================
def apply_traffic_conditions(G, time_label):
    """Applies simulated congestion and penalties to the graph edges."""
    all_edges = list(G.edges(data=True))
    
    # Configure congestion based on scenario
    if time_label == "Morning_Peak":
        congestion_rate = 0.20; primary_weight = 90
    elif time_label == "Evening_Peak":
        congestion_rate = 0.15; primary_weight = 70
    else: 
        congestion_rate = 0.05; primary_weight = 10

    # Determine edge weights for obstacle selection (prioritize major roads for blocking)
    weights = []
    for u, v, data in all_edges:
        hw_type = str(clean_attribute(data.get('highway'), 'unclassified'))
        if 'primary' in hw_type: w = primary_weight
        elif 'secondary' in hw_type: w = primary_weight * 0.6
        elif 'tertiary' in hw_type: w = 30
        else: w = 5
        weights.append(w)
            
    # Select which roads are "congested" (obstacles)
    num_obstacles = int(len(all_edges) * congestion_rate)
    obstacle_keys = set()
    if num_obstacles > 0:
        selected_indices = random.choices(range(len(all_edges)), weights=weights, k=num_obstacles * 2)
        obstacle_edges = [all_edges[i] for i in list(set(selected_indices))[:num_obstacles]]
        obstacle_keys = set([(u, v) for u, v, d in obstacle_edges])

    # Assign traffic status and shortest-path weight
    for u, v, data in G.edges(data=True):
        data['time_scenario'] = time_label 
        if (u, v) in obstacle_keys:
            data['traffic_status'] = "Congested"
            data['is_obstacle'] = 1     
            data['weight'] = 9999999    # Heavy penalty forces pathfinder to avoid this road
        else:
            data['traffic_status'] = "Clear"
            data['is_obstacle'] = 0
            # Weight is the travel time (seconds) for clear roads
            data['weight'] = data.get('travel_time', data['length']) 
            
    return G

# ==========================================
# 4. Data Generation (Dual Datasets)
# ==========================================
def generate_all_datasets(G_base):
    print("📊 Generating Datasets...")
    training_data = [] # Dataset B (ML Training)
    map_status_data = [] # Dataset A (Map Status)
    
    scenarios = ["Morning_Peak", "Off_Peak", "Evening_Peak"]
    
    for time_period in scenarios:
        G_scenario = G_base.copy()
        G_scenario = apply_traffic_conditions(G_scenario, time_period)
        
        # --- [Dataset A] Collect Map Status ---
        print(f"   -> Snapshotting Map Status for {time_period}...")
        for u, v, d in G_scenario.edges(data=True):
            raw_type = str(clean_attribute(d.get('highway'), 'unclassified'))
            map_status_data.append({
                "Scenario": time_period,
                "Road_ID": f"{u}-{v}",
                "Road_Name": clean_attribute(d.get('name'), "Unnamed"),
                "Road_Type": categorize_road(raw_type),
                "Status": d['traffic_status'], 
                "Is_Blocked": d['is_obstacle'],
                "Length": float(f"{d['length']:.1f}")
            })

        # --- [Dataset B] Collect Training Data ---
        nodes = list(G_scenario.nodes())
        print(f"   -> Simulating Trips for {time_period}...")
        
        # Function to extract a single row for the training dataset
        def extract_row(u, v, d, label):
            raw_type = str(clean_attribute(d.get('highway'), 'unclassified'))
            distance_m = d['length']
            
            # Use the original travel_time calculated by OSMPnx
            time_taken_sec = d.get('travel_time', distance_m / (40 / 3.6)) 
            
            # Calculate average speed (Distance / Time)
            avg_speed_kph = (distance_m / time_taken_sec) * 3.6 
            
            return {
                "timestamp": generate_fake_timestamp(time_period),
                # Removed "start" and "end" to simplify the dataset
                "distance": float(f"{distance_m:.1f}"),
                "intersection": G_scenario.degree[u],
                "traffic": d['traffic_status'],
                "obstacle": int(d['is_obstacle']),
                "time_taken": float(f"{time_taken_sec:.2f}"), # Time (time taken)
                "score": float(f"{d['weight']:.2f}"),
                "is_chosen": label,
                "avg_speed": float(f"{avg_speed_kph:.1f}"),   # Speed (average speed)
                "Road_type": categorize_road(raw_type),
                "One_way": "Yes" if d.get('oneway') else "No"
            }

        # Simulate trips and generate YES/NO choices
        for i in range(150): 
            start, end = random.sample(nodes, 2)
            try:
                # Find the optimal path based on 'weight' (which avoids congestion)
                path = nx.shortest_path(G_scenario, start, end, weight='weight')
                path_edges = set(zip(path[:-1], path[1:]))
                
                # Positive Examples (Chosen)
                for u, v in path_edges:
                    d = G_scenario.get_edge_data(u, v)[0]
                    # Pass 'end' to extract_row if needed, but not strictly required here
                    training_data.append(extract_row(u, v, d, "YES"))
                
                # Negative Examples (Not Chosen)
                # Sample 3 random edges not on the path
                sample_neg = random.sample(list(G_scenario.edges()), 3)
                for u, v in sample_neg:
                    if (u, v) not in path_edges:
                        d = G_scenario.get_edge_data(u, v)[0]
                        # Pass 'end' to extract_row if needed
                        training_data.append(extract_row(u, v, d, "NO"))
            except nx.NetworkXNoPath: continue
                
    return training_data, map_status_data

# ==========================================
# 5. Interactive Map Logic (No changes needed here)
# ==========================================
class InteractiveRoutePlanner:
    def __init__(self, G, time_label):
        self.G = G
        self.time_label = time_label
        self.start_node = None; self.end_node = None
        self.start_name = ""; self.end_name = ""
        self.landmark_labels = [] 
        self.scatter_start = None; self.scatter_end = None
        self.text_start = None; self.text_end = None
        self.route_lines = [] 
        
        print("   -> Rendering map...")
        edge_colors = []
        edge_widths = []
        for u, v, d in G.edges(data=True):
            if d.get('is_obstacle') == 1:
                edge_colors.append('#FF0000'); edge_widths.append(2.0) # Red for Congested
            else:
                hw_type = str(clean_attribute(d.get('highway'), ''))
                if 'motorway' in hw_type or 'trunk' in hw_type:
                    edge_colors.append('#E65100'); edge_widths.append(1.5)
                elif 'primary' in hw_type:
                    edge_colors.append('#FFA000'); edge_widths.append(1.2)
                elif 'secondary' in hw_type:
                    edge_colors.append('#FBC02D'); edge_widths.append(1.0)
                else:
                    edge_colors.append('#BDBDBD'); edge_widths.append(0.5)

        self.fig, self.ax = ox.plot_graph(G, edge_color=edge_colors, edge_linewidth=edge_widths, 
                                          node_size=0, show=False, bgcolor='white')
        
        self.ax.callbacks.connect('xlim_changed', self.on_xlim_changed)
        self.default_title = f"[{time_label}] Left-Click: Select Points | Right-Click: Reset All"
        self.ax.set_title(self.default_title, fontsize=10)
        
        try:
            if hasattr(self.fig.canvas, 'manager'):
                self.fig.canvas.manager.set_window_title(f"Penang Traffic - {time_label}")
        except: pass 
            
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def on_xlim_changed(self, ax):
        pass

    def reset_selection(self):
        self.start_node = None; self.end_node = None
        if self.scatter_start: self.scatter_start.remove(); self.scatter_start = None
        if self.scatter_end: self.scatter_end.remove(); self.scatter_end = None
        if self.text_start: self.text_start.remove(); self.text_start = None
        if self.text_end: self.text_end.remove(); self.text_end = None
        for line in self.route_lines:
            try: line.remove() 
            except: pass
        self.route_lines = []
        self.ax.set_title(self.default_title, fontsize=10, color='black')
        self.fig.canvas.draw()

    def onclick(self, event):
        if self.fig.canvas.toolbar.mode != "":
            self.ax.set_title(f"⚠️ ZOOM TOOL ACTIVE! Turn it off to select.", color='red')
            self.fig.canvas.draw(); return
        if event.xdata is None or event.ydata is None: return
        if event.button == 3: self.reset_selection(); return

        if self.start_node is None:
            self.start_node = ox.nearest_nodes(self.G, event.xdata, event.ydata)
            self.scatter_start = self.ax.scatter(event.xdata, event.ydata, c='green', s=100, zorder=5, edgecolors='black')
            self.ax.set_title(f"Start Set. Click for END point.", fontsize=10)
            self.fig.canvas.draw()
        elif self.end_node is None:
            self.end_node = ox.nearest_nodes(self.G, event.xdata, event.ydata)
            self.scatter_end = self.ax.scatter(event.xdata, event.ydata, c='red', s=100, zorder=5, edgecolors='black')
            self.ax.set_title(f"Calculating...", fontsize=10, color='blue')
            self.fig.canvas.draw()
            self.calculate_route()

    def calculate_route(self):
        try:
            # Shortest path is based on 'weight' which avoids blocked roads
            route = nx.shortest_path(self.G, self.start_node, self.end_node, weight='weight')
            # Plot the route on the map
            ox.plot_graph_route(self.G, route, route_color='blue', route_linewidth=4, ax=self.ax, show=False)
            
            # Capture route lines for reset
            current_collections = self.ax.collections[-1:]
            self.route_lines.extend(current_collections)

            # Save a snapshot of the route
            path_overview = os.path.join(TARGET_DIR, f"route_overview_{self.time_label}.png")
            self.fig.savefig(path_overview, dpi=300, bbox_inches='tight')
            print(f"📸 Saved Overview: {path_overview}")

            self.ax.set_title(f"Route Found! (Right-Click to Reset)", fontsize=10, color='black')
            self.fig.canvas.draw()
            
        except nx.NetworkXNoPath:
            self.ax.set_title("Blocked by Traffic!", fontsize=12, color='red')
            self.fig.canvas.draw()
        except Exception as e: print(f"Error: {e}")

# ==========================================
# 6. GUI Launcher and Main Execution
# ==========================================
def start_launcher_gui(G_base):
    root = tk.Tk()
    root.title("AI Traffic Navigator")
    root.geometry("400x250")
    
    tk.Label(root, text="🚦 Penang Smart Traffic AI", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(root, text="Select Scenario:", font=("Arial", 10)).pack(pady=5)

    time_var = tk.StringVar(value="Morning_Peak")
    time_combo = ttk.Combobox(root, textvariable=time_var, width=40, state="readonly")
    time_combo['values'] = ("Morning_Peak", "Evening_Peak", "Off_Peak")
    time_combo.pack(pady=5)
    
    def on_launch():
        time_label = time_var.get()
        print(f"\n🚀 Launching Interactive Map for [{time_label}]...")
        G_sim = G_base.copy()
        # Apply traffic for the selected scenario
        G_sim = apply_traffic_conditions(G_sim, time_label) 
        root.destroy()
        InteractiveRoutePlanner(G_sim, time_label)

    btn_launch = ttk.Button(root, text="Open Map & Click to Navigate", command=on_launch)
    btn_launch.pack(pady=20)
    root.mainloop()

def main():
    if not os.path.exists(TARGET_DIR):
        try: os.makedirs(TARGET_DIR)
        except: pass

    print(f"⬇️  Initializing Map...")
    # Exclude major highways for a local network focus
    cf = '["highway"!~"motorway|trunk|motorway_link|trunk_link"]' 
    G = ox.graph_from_point(TARGET_LOCATION, dist=SEARCH_RADIUS, custom_filter=cf, network_type='drive')
    
    G = ox.add_edge_bearings(G) 
    G = ox.add_edge_speeds(G, fallback=40) 
    # Ensure attributes exist before calculating travel time
    for u, v, d in G.edges(data=True):
        if 'speed_kph' not in d or pd.isna(d['speed_kph']): d['speed_kph'] = 40.0
        if 'length' not in d or pd.isna(d['length']): d['length'] = 50.0
    G = ox.add_edge_travel_times(G) # This calculates 'travel_time' (seconds)
    
    # Keep only the largest connected component for reliable pathfinding
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G_base = G.subgraph(largest_cc).copy()
    print("✅ Map Ready.")

    # --- Generate BOTH Datasets ---
    training_data, map_status_data = generate_all_datasets(G_base)
    
    # 1. Save Training Data (Dataset B)
    csv_train = os.path.join(TARGET_DIR, "penang_training_data.csv")
    df_train = pd.DataFrame(training_data)
    
    # FINAL COLUMN LIST matching your request:
    cols = ["timestamp", "distance", "intersection", "traffic", 
            "obstacle", "time_taken", "score", "is_chosen", 
            "avg_speed", "Road_type", "One_way"]
            
    df_train = df_train[cols]
    df_train.to_csv(csv_train, index=False)
    print(f"✅ Saved Training Data (Dataset B): {csv_train}")

    # 2. Save Map Status Data (Dataset A)
    csv_status = os.path.join(TARGET_DIR, "penang_map_status.csv")
    df_status = pd.DataFrame(map_status_data)
    df_status.to_csv(csv_status, index=False)
    print(f"✅ Saved Map Status Data (Dataset A): {csv_status}")
    
    print("🖥️  Starting GUI...")
    start_launcher_gui(G_base)

if __name__ == "__main__":
    main()