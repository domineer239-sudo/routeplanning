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
# Use raw string for Windows paths to avoid escape character issues
TARGET_DIR = r"D:\202509的课件\ai作业"
# Coordinates for George Town, Penang (Heritage Zone)
TARGET_LOCATION = (5.4195, 100.3325) 
SEARCH_RADIUS = 1200 

# ==========================================
# 2. Helper Functions
# ==========================================
def get_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing between two points."""
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
    """Cleans OSM attributes which might be lists or nan."""
    if isinstance(val, list): return val[0] 
    if pd.isna(val) or val is None: return default_val
    return val

# ==========================================
# 3. Traffic Simulation Logic
# ==========================================
def apply_traffic_conditions(G, time_label):
    """Applies traffic congestion based on the time of day."""
    all_edges = list(G.edges(data=True))
    
    # Define parameters for different time periods
    if time_label == "Morning_Peak":
        congestion_rate = 0.20
        primary_weight = 90 # High probability for main roads to be blocked
    elif time_label == "Evening_Peak":
        congestion_rate = 0.15
        primary_weight = 70
    else: # Off_Peak
        congestion_rate = 0.05
        primary_weight = 10

    # Calculate weights for random sampling (Probabilistic Traffic)
    weights = []
    for u, v, data in all_edges:
        hw_type = str(clean_attribute(data.get('highway'), 'unclassified'))
        
        # Main roads are more likely to be congested during peaks
        if hw_type in ['primary', 'primary_link']: w = primary_weight
        elif hw_type in ['secondary', 'secondary_link']: w = primary_weight * 0.6
        elif hw_type in ['tertiary']: w = 30
        else: w = 5 # Residential roads are less likely to be congested
        weights.append(w)
        
    # Generate obstacles (Congestion)
    num_obstacles = int(len(all_edges) * congestion_rate)
    obstacle_keys = set()
    if num_obstacles > 0:
        # Weighted random selection
        selected_indices = random.choices(range(len(all_edges)), weights=weights, k=num_obstacles * 2)
        # Remove duplicates and slice
        obstacle_edges = [all_edges[i] for i in list(set(selected_indices))[:num_obstacles]]
        obstacle_keys = set([(u, v) for u, v, d in obstacle_edges])

    # Update Graph Attributes
    for u, v, data in G.edges(data=True):
        if (u, v) in obstacle_keys:
            data['is_obstacle'] = 1     
            data['weight'] = 9999999  # Infinite cost for blocked roads  
        else:
            data['is_obstacle'] = 0
            # Normal cost is travel time
            data['weight'] = data.get('travel_time', data['length'])
            
    return G

# ==========================================
# 4. Data Generation (For Weka)
# ==========================================
def generate_training_data(G_base):
    print("📊 Generating Weka training data (this may take a moment)...")
    all_data = []
    scenarios = ["Morning_Peak", "Off_Peak", "Evening_Peak"]
    
    for time_period in scenarios:
        # Create a fresh copy of the map for this scenario
        G_scenario = G_base.copy()
        G_scenario = apply_traffic_conditions(G_scenario, time_period)
        
        nodes = list(G_scenario.nodes())
        print(f"   -> Processing scenario: {time_period}...")
        
        # Simulate 100 trips per scenario
        for i in range(100): 
            start, end = random.sample(nodes, 2)
            try:
                # Get destination data for bearing calculation
                end_node_data = G_scenario.nodes[end]
                
                # Expert pathfinding (A* / Dijkstra)
                path = nx.shortest_path(G_scenario, start, end, weight='weight')
                path_edges = set(zip(path[:-1], path[1:]))
                
                # Function to extract features for a single road segment
                def extract_row(u, v, d, label):
                    road_bearing = d.get('bearing', 0)
                    u_node = G_scenario.nodes[u]
                    target_bearing = get_bearing(u_node['y'], u_node['x'], end_node_data['y'], end_node_data['x'])
                    angle_diff = abs(road_bearing - target_bearing)
                    if angle_diff > 180: angle_diff = 360 - angle_diff
                    
                    return {
                        "Time_Period": time_period,
                        "Road_Type": str(clean_attribute(d.get('highway'), 'unclassified')),
                        "Max_Speed": float(clean_attribute(d.get('maxspeed'), 40)),
                        "Lanes": int(clean_attribute(d.get('lanes'), 1)),
                        "Road_Length": float(d['length']),
                        "Angle_Deviation": float(f"{angle_diff:.2f}"),
                        "One_Way": str(1 if d.get('oneway') else 0),
                        "Is_Obstacle": int(d['is_obstacle']),
                        "Should_Take": label
                    }

                # Positive Samples (The expert path)
                for u, v in path_edges:
                    d = G_scenario.get_edge_data(u, v)[0]
                    all_data.append(extract_row(u, v, d, "YES"))
                    
                # Negative Samples (Random roads not taken)
                sample_neg = random.sample(list(G_scenario.edges()), 2)
                for u, v in sample_neg:
                    if (u, v) not in path_edges:
                        d = G_scenario.get_edge_data(u, v)[0]
                        all_data.append(extract_row(u, v, d, "NO"))

            except nx.NetworkXNoPath:
                continue
                
    return all_data

# ==========================================
# 5. Interactive Map Logic
# ==========================================
class InteractiveRoutePlanner:
    def __init__(self, G, time_label):
        self.G = G
        self.time_label = time_label
        self.start_node = None
        self.end_node = None
        
        # Setup the plot
        print("   -> Rendering map for interaction...")
        # Obstacles are red, normal roads are grey
        ec = ['#ff0000' if d.get('is_obstacle')==1 else '#999999' for u, v, d in G.edges(data=True)]
        # Obstacles are thicker
        ew = [2.0 if d.get('is_obstacle')==1 else 0.5 for u, v, d in G.edges(data=True)]
        
        self.fig, self.ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=ew, node_size=0, 
                                          show=False, bgcolor='white')
        
        # [FIXED] Updated set_window_title call for newer matplotlib versions
        # 修复：使用 manager 设置标题，并增加容错处理
        try:
            if hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager:
                self.fig.canvas.manager.set_window_title(f"Penang Traffic Simulator - {time_label}")
            elif hasattr(self.fig.canvas, 'set_window_title'):
                self.fig.canvas.set_window_title(f"Penang Traffic Simulator - {time_label}")
        except Exception:
            # Fallback: 如果都失败了，就不设置窗口标题，保证程序不崩溃
            pass 
            
        self.ax.set_title(f"[{time_label}] Click map to select START point", fontsize=12, color='green')
        
        # Connect the click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        # Ignore clicks outside the axes
        if event.xdata is None or event.ydata is None:
            return

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
            
            # 3. Calculate and Draw Route
            self.calculate_route()

    def calculate_route(self):
        try:
            route = nx.shortest_path(self.G, self.start_node, self.end_node, weight='weight')
            
            # Plot the route
            ox.plot_graph_route(self.G, route, route_color='blue', route_linewidth=4, 
                                ax=self.ax, show=False)
            
            self.ax.set_title(f"[{self.time_label}] Route Found! (Length: {len(route)} steps)", fontsize=12, color='black')
            self.fig.canvas.draw()
            print("✅ Route calculation successful!")
            
            # Save the result
            img_path = os.path.join(TARGET_DIR, f"interactive_result_{self.time_label}.png")
            self.fig.savefig(img_path, dpi=300, bbox_inches='tight')
            print(f"💾 Route image saved to: {img_path}")
            
            # Disable further clicks
            self.fig.canvas.mpl_disconnect(self.cid)
            
        except nx.NetworkXNoPath:
            self.ax.set_title("No path found! (Blocked by traffic)", fontsize=12, color='red')
            self.fig.canvas.draw()
            print("❌ No path found due to obstacles.")
        except Exception as e:
            print(f"Error: {e}")

# ==========================================
# 6. GUI Entry Point
# ==========================================
def start_launcher_gui(G_base):
    root = tk.Tk()
    root.title("AI Traffic Navigator (Penang)")
    root.geometry("400x300")
    
    # Styling
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 11), padding=10)
    
    tk.Label(root, text="🚦 Penang Smart Traffic AI", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(root, text="Select Time Scenario:", font=("Arial", 10)).pack(pady=5)

    # Time Selection
    time_var = tk.StringVar(value="Morning_Peak")
    time_combo = ttk.Combobox(root, textvariable=time_var, width=30, state="readonly")
    time_combo['values'] = ("Morning_Peak", "Evening_Peak", "Off_Peak")
    time_combo.pack(pady=5)
    
    def on_launch():
        time_label = time_var.get()
        print(f"\n🚀 Launching Interactive Map for [{time_label}]...")
        
        # Apply traffic to a copy of the graph
        G_sim = G_base.copy()
        G_sim = apply_traffic_conditions(G_sim, time_label)
        
        # Start Matplotlib Interactive Mode
        root.destroy() # Close the launcher window
        InteractiveRoutePlanner(G_sim, time_label)

    btn = ttk.Button(root, text="Open Map to Select Route", command=on_launch)
    btn.pack(pady=20)
    
    tk.Label(root, text="Instructions:\n1. Select time period.\n2. Click button to open map.\n3. Click map twice (Start & End).", 
             font=("Arial", 9), fg="gray", justify="center").pack(side=tk.BOTTOM, pady=20)

    root.mainloop()

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
    
    # [FIX] Data Cleaning MUST happen BEFORE adding edge travel times
    print("   -> Cleaning data (filling missing speeds)...")
    for u, v, d in G.edges(data=True):
        if 'speed_kph' not in d or pd.isna(d['speed_kph']): d['speed_kph'] = 40.0
        if 'length' not in d or pd.isna(d['length']): d['length'] = 50.0

    # Now it is safe to calculate travel times
    G = ox.add_edge_travel_times(G)
    
    # Keep largest component
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G_base = G.subgraph(largest_cc).copy()
    print("✅ Map Initialized!")

    # Generate CSV if missing
    csv_path = os.path.join(TARGET_DIR, "penang_traffic_data_english.csv")
    if not os.path.exists(csv_path):
        data = generate_training_data(G_base)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"✅ Training data generated: {csv_path}")
    else:
        print("✅ Training data found, skipping generation.")

    # Start the GUI
    print("🖥️  Starting GUI...")
    start_launcher_gui(G_base)

if __name__ == "__main__":
    main()