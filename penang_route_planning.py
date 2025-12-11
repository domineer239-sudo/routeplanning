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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration
# ==========================================
# Use raw string for Windows paths
TARGET_DIR = r"D:\202509的课件\ai作业"
# Coordinates for George Town, Penang (Heritage Zone)
TARGET_LOCATION = (5.4195, 100.3325) 
SEARCH_RADIUS = 1200 

# ==========================================
# 2. Helper Functions & Math Tools
# ==========================================
def get_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing (angle) between two points."""
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    brng = math.atan2(y, x)
    brng = math.degrees(brng)
    return (brng + 360) % 360

def clean_attribute(val, default_val):
    """Cleans OSM attributes which might be lists or NaN."""
    if isinstance(val, list): return val[0] 
    if pd.isna(val) or val is None: return default_val
    return val

def categorize_road(osm_type):
    """Categorizes complex OSM road types into human-readable hierarchy."""
    t = str(osm_type)
    if 'motorway' in t or 'trunk' in t: return "Highway"
    if 'primary' in t: return "Main_Artery"
    if 'secondary' in t: return "Secondary_Road"
    if 'tertiary' in t: return "Connector_Road"
    if 'residential' in t or 'living_street' in t: return "Local_Street"
    return "Small_Alley"

def generate_timestamp(scenario):
    """Generates a random timestamp string based on the scenario."""
    if scenario == "Morning_Peak":
        hour = random.randint(7, 9)
    elif scenario == "Evening_Peak":
        hour = random.randint(17, 19)
    else: # Off_Peak
        hour = random.choice([random.randint(10, 16), random.randint(21, 23)])
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    # Using a fixed date for consistency
    return f"2024-12-15 {hour:02d}:{minute:02d}:{second:02d}"

# ==========================================
# 3. Traffic Simulation Core (Logic)
# ==========================================
def apply_traffic_conditions(G, time_label):
    """Applies traffic congestion based on the time of day."""
    all_edges = list(G.edges(data=True))
    
    # Define parameters based on time scenario
    if time_label == "Morning_Peak":
        congestion_rate = 0.20
        primary_weight = 90  # Main roads are very likely to be blocked
    elif time_label == "Evening_Peak":
        congestion_rate = 0.15
        primary_weight = 70
    else: # Off_Peak
        congestion_rate = 0.05
        primary_weight = 10  # Main roads are clear at night

    # Calculate probability weights for congestion
    weights = []
    for u, v, data in all_edges:
        hw_type = str(clean_attribute(data.get('highway'), 'unclassified'))
        if hw_type in ['primary', 'primary_link']: w = primary_weight
        elif hw_type in ['secondary', 'secondary_link']: w = primary_weight * 0.6
        elif hw_type in ['tertiary']: w = 30
        else: w = 5 # Local streets are less likely to be congested
        weights.append(w)
        
    # Generate random obstacles (Simulating real traffic jams)
    num_obstacles = int(len(all_edges) * congestion_rate)
    obstacle_keys = set()
    if num_obstacles > 0:
        selected_indices = random.choices(range(len(all_edges)), weights=weights, k=num_obstacles * 2)
        obstacle_edges = [all_edges[i] for i in list(set(selected_indices))[:num_obstacles]]
        obstacle_keys = set([(u, v) for u, v, d in obstacle_edges])

    # Update Graph Attributes
    for u, v, data in G.edges(data=True):
        if (u, v) in obstacle_keys:
            data['traffic_status'] = "Congested" 
            data['is_obstacle'] = 1     
            data['weight'] = 9999999  # Infinite cost
        else:
            data['traffic_status'] = "Clear"
            data['is_obstacle'] = 0
            data['weight'] = data.get('travel_time', data['length'])
            
    return G

# ==========================================
# 4. Dataset Generation (Requested 13 Columns)
# ==========================================
def generate_training_data(G_base):
    print("📊 Generating dataset with specified 13 columns...")
    all_data = []
    scenarios = ["Morning_Peak", "Off_Peak", "Evening_Peak"]
    
    for time_period in scenarios:
        G_scenario = G_base.copy()
        G_scenario = apply_traffic_conditions(G_scenario, time_period)
        
        nodes = list(G_scenario.nodes())
        print(f"   -> Processing Scenario: {time_period}...")
        
        # Simulate 150 trips per scenario
        for i in range(150): 
            start_node, end_node = random.sample(nodes, 2)
            try:
                # Expert Pathfinding (Ground Truth)
                path = nx.shortest_path(G_scenario, start_node, end_node, weight='weight')
                path_edges = set(zip(path[:-1], path[1:]))
                
                # Internal helper to extract features
                def extract_row(u, v, d, label):
                    # Data preparation
                    raw_type = str(clean_attribute(d.get('highway'), 'unclassified'))
                    
                    # Calculate Intersection Complexity (Degree of the start node)
                    # How many roads connect to this intersection?
                    intersection_complexity = G_scenario.degree[u]
                    
                    return {
                        "timestamp": generate_timestamp(time_period), # 1. timestamp
                        "start": u,                                   # 2. start (Node ID)
                        "end": v,                                     # 3. end (Node ID)
                        "distance": float(f"{d['length']:.1f}"),      # 4. distance
                        "intersection": intersection_complexity,      # 5. intersection (Degree)
                        "traffic": d['traffic_status'],               # 6. traffic (Clear/Congested)
                        "obstacle": int(d['is_obstacle']),            # 7. obstacle (0/1)
                        "time": time_period,                          # 8. time (Scenario)
                        "score": float(f"{d['weight']:.2f}"),         # 9. score (Cost/Weight)
                        "is_chosen": label,                           # 10. is_chosen (YES/NO)
                        "Speed": float(clean_attribute(d.get('maxspeed'), 40)), # 11. Speed
                        "Road_type": categorize_road(raw_type),       # 12. Road_type
                        "One_way": "Yes" if d.get('oneway') else "No" # 13. One_way
                    }

                # Positive Samples (Expert Path)
                for u, v in path_edges:
                    d = G_scenario.get_edge_data(u, v)[0]
                    all_data.append(extract_row(u, v, d, "YES"))
                    
                # Negative Samples (Random roads not taken)
                sample_neg = random.sample(list(G_scenario.edges()), 3)
                for u, v in sample_neg:
                    if (u, v) not in path_edges:
                        d = G_scenario.get_edge_data(u, v)[0]
                        all_data.append(extract_row(u, v, d, "NO"))

            except nx.NetworkXNoPath:
                continue
                
    return all_data

# ==========================================
# 5. Interactive Map Logic (GUI)
# ==========================================
class InteractiveRoutePlanner:
    def __init__(self, G, time_label):
        self.G = G
        self.time_label = time_label
        self.start_node = None
        self.end_node = None
        
        print("   -> Rendering map for interaction...")
        # Obstacles are red, normal roads are grey
        ec = ['#ff0000' if d.get('is_obstacle')==1 else '#999999' for u, v, d in G.edges(data=True)]
        ew = [2.0 if d.get('is_obstacle')==1 else 0.5 for u, v, d in G.edges(data=True)]
        
        self.fig, self.ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=ew, node_size=0, 
                                          show=False, bgcolor='white')
        
        # Window Title Compatibility Fix
        try:
            if hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager:
                self.fig.canvas.manager.set_window_title(f"Penang Traffic Simulator - {time_label}")
            elif hasattr(self.fig.canvas, 'set_window_title'):
                self.fig.canvas.set_window_title(f"Penang Traffic Simulator - {time_label}")
        except Exception:
            pass 
            
        self.ax.set_title(f"[{time_label}] Click map to select START point", fontsize=12, color='green')
        
        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        if event.xdata is None or event.ydata is None: return

        # 1. Select Start Point
        if self.start_node is None:
            self.start_node = ox.nearest_nodes(self.G, event.xdata, event.ydata)
            self.ax.scatter(event.xdata, event.ydata, c='green', s=100, label='Start', zorder=5, edgecolors='black')
            self.ax.set_title(f"[{self.time_label}] Start set! Now click to select END point", fontsize=12, color='red')
            self.fig.canvas.draw()
            print(f"📍 Start Point Selected: Node {self.start_node}")
            
        # 2. Select End Point
        elif self.end_node is None:
            self.end_node = ox.nearest_nodes(self.G, event.xdata, event.ydata)
            if self.end_node == self.start_node:
                print("⚠️ Start and End are too close!")
                return

            self.ax.scatter(event.xdata, event.ydata, c='red', s=100, label='End', zorder=5, edgecolors='black')
            self.ax.set_title(f"[{self.time_label}] Calculating Route...", fontsize=12, color='blue')
            self.fig.canvas.draw()
            print(f"🏁 End Point Selected: Node {self.end_node}")
            self.calculate_route()

    def calculate_route(self):
        try:
            route = nx.shortest_path(self.G, self.start_node, self.end_node, weight='weight')
            
            ox.plot_graph_route(self.G, route, route_color='blue', route_linewidth=4, 
                                ax=self.ax, show=False)
            
            self.ax.set_title(f"[{self.time_label}] Route Found! (Length: {len(route)} segments)", fontsize=12, color='black')
            self.fig.canvas.draw()
            print("✅ Route calculation successful!")
            
            # Save the result image
            img_path = os.path.join(TARGET_DIR, f"interactive_result_{self.time_label}.png")
            self.fig.savefig(img_path, dpi=300, bbox_inches='tight')
            print(f"💾 Route image saved to: {img_path}")
            
            self.fig.canvas.mpl_disconnect(self.cid)
            
        except nx.NetworkXNoPath:
            self.ax.set_title("No path found! (Blocked by traffic)", fontsize=12, color='red')
            self.fig.canvas.draw()
            print("❌ No path found due to obstacles.")
        except Exception as e:
            print(f"Error: {e}")

# ==========================================
# 6. GUI Launcher
# ==========================================
def start_launcher_gui(G_base):
    root = tk.Tk()
    root.title("AI Traffic Navigator (Penang)")
    root.geometry("400x350")
    
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 11), padding=10)
    
    tk.Label(root, text="🚦 Penang Smart Traffic AI", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(root, text="Select Time Scenario:", font=("Arial", 10)).pack(pady=5)

    time_var = tk.StringVar(value="Morning_Peak")
    time_combo = ttk.Combobox(root, textvariable=time_var, width=30, state="readonly")
    time_combo['values'] = ("Morning_Peak", "Evening_Peak", "Off_Peak")
    time_combo.pack(pady=5)
    
    def on_launch():
        time_label = time_var.get()
        print(f"\n🚀 Launching Interactive Map for [{time_label}]...")
        
        G_sim = G_base.copy()
        G_sim = apply_traffic_conditions(G_sim, time_label)
        
        root.destroy()
        InteractiveRoutePlanner(G_sim, time_label)

    btn = ttk.Button(root, text="Open Map to Select Route", command=on_launch)
    btn.pack(pady=20)
    
    tk.Label(root, text="Instructions:\n1. Select time period.\n2. Click button to open map.\n3. Click map twice (Start & End).", 
             font=("Arial", 9), fg="gray", justify="center").pack(side=tk.BOTTOM, pady=20)

    root.mainloop()

# ==========================================
# 7. Main Execution
# ==========================================
def main():
    if not os.path.exists(TARGET_DIR):
        try: os.makedirs(TARGET_DIR)
        except: pass

    print(f"⬇️  Initializing Map Data for {TARGET_LOCATION}...")
    # Filter out motorways to focus on city streets
    cf = '["highway"!~"motorway|trunk|motorway_link|trunk_link"]'
    G = ox.graph_from_point(TARGET_LOCATION, dist=SEARCH_RADIUS, custom_filter=cf, network_type='drive')
    
    # Pre-process graph
    G = ox.add_edge_bearings(G) 
    G = ox.add_edge_speeds(G, fallback=40) 
    
    # Data Cleaning (MUST be done before travel times)
    print("   -> Cleaning data (filling missing attributes)...")
    for u, v, d in G.edges(data=True):
        if 'speed_kph' not in d or pd.isna(d['speed_kph']): d['speed_kph'] = 40.0
        if 'length' not in d or pd.isna(d['length']): d['length'] = 50.0

    G = ox.add_edge_travel_times(G)
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G_base = G.subgraph(largest_cc).copy()
    print("✅ Map Initialized!")

    # Generate CSV (Force overwrite to penang_professional_data.csv)
    csv_path = os.path.join(TARGET_DIR, "penang_professional_data.csv")
    print(f"♻️  Generating fresh dataset: {csv_path}")
    data = generate_training_data(G_base)
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print(f"\n🎉 Success! Dataset generated with {len(df)} rows.")
    print("Columns: timestamp, start, end, distance, intersection, traffic, obstacle, time, score, is_chosen, Speed, Road_type, One_way")
    
    # Start the GUI
    print("🖥️  Starting GUI...")
    start_launcher_gui(G_base)

if __name__ == "__main__":
    main()
    
    print("\n🎉 Success! Dataset created with columns:")
    print("1. Time_Scenario | 2. Road_Hierarchy | 3. Traffic_Condition")
    print("4. Flow_Direction | 5. Lane_Count | 6. Angle_Deviation | 7. Decision")
    
    # Start the GUI
    print("🖥️  Starting GUI...")
    start_launcher_gui(G_base)

if __name__ == "__main__":
    main()