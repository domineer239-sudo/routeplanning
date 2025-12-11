import osmnx as ox
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import math
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.path import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration
# ==========================================
# Use raw string for Windows paths to avoid escape character issues
# 请确保这个路径存在，或者代码会自动创建它
TARGET_DIR = r"C:\Users\Admin\Desktop\XMUM\DSC\2509\Principles of Artificial Intelligence\AI assignment\ai group"
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
# 4. Data Generation (For Weka) - UPDATED
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
                # [MODIFIED] 仅修改此函数，加入 Max_Speed 估算逻辑
                def extract_row(u, v, d, label):
                    road_bearing = d.get('bearing', 0)
                    u_node = G_scenario.nodes[u]
                    
                    # Calculate angle deviation
                    target_bearing = get_bearing(u_node['y'], u_node['x'], end_node_data['y'], end_node_data['x'])
                    angle_diff = abs(road_bearing - target_bearing)
                    if angle_diff > 180: angle_diff = 360 - angle_diff
                    
                    road_type = str(clean_attribute(d.get('highway'), 'unclassified'))
                    
                    # --- [关键修改：Max_Speed 估算逻辑] ---
                    raw_speed = d.get('maxspeed')
                    
                    # 1. 使用原生数据 (并尝试清洗)
                    final_speed = 40.0 # 默认保底值
                    if raw_speed and not pd.isna(raw_speed):
                        if isinstance(raw_speed, list): raw_speed = raw_speed[0]
                        try:
                            # 尝试提取数字，例如 "50 km/h" -> 50
                            final_speed = float(str(raw_speed).split()[0])
                        except:
                            pass # 转换失败，使用 40.0 保底
                    
                    # 2. 如果数据缺失（或转换失败），根据道路类型进行估算
                    if final_speed <= 40.0:
                        if road_type in ['primary', 'primary_link', 'trunk']:
                            final_speed = 70.0
                        elif road_type in ['secondary', 'secondary_link']:
                            final_speed = 60.0
                        elif road_type in ['tertiary']:
                            final_speed = 50.0
                        elif road_type in ['residential', 'living_street']:
                            final_speed = 30.0
                        else:
                            final_speed = 40.0 
                    # ------------------------------------
                    
                    # --- [Travel Time Base 特征] ---
                    # The ideal travel time (used for the FEATURE, derived from graph's time if available)
                    base_time = d.get('travel_time')
                    if base_time is None or pd.isna(base_time):
                        # 如果 graph 原始 travel_time 缺失，则使用我们估算的 final_speed 来计算 base_time
                        speed_mps = final_speed / 3.6 
                        base_time = float(d['length']) / max(speed_mps, 1.0)
                    
                    # --- [Node Degree 特征] ---
                    # How many roads connect to the starting intersection (complexity)
                    degree = len(list(G_scenario.neighbors(u)))

                    return {
                        "Time_Period": time_period,
                        "Road_Type": road_type,
                        "Max_Speed": final_speed,                          # <-- 使用估算后的速度作为 CSV 特征
                        "Lanes": int(clean_attribute(d.get('lanes'), 1)),
                        "Road_Length": float(d['length']),
                        "Angle_Deviation": float(f"{angle_diff:.2f}"),
                        "One_Way": str(1 if d.get('oneway') else 0),
                        "Is_Obstacle": int(d['is_obstacle']),
                        "Travel_Time_Base": float(f"{base_time:.2f}"),
                        "Node_Degree": int(degree),
                        "Should_Take": label                           # Class Attribute
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
# 5. Interactive Map Logic (No change)
# ==========================================
class InteractiveRoutePlanner:
    def __init__(self, G, time_label):
        self.G = G
        self.time_label = time_label
        self.start_node = None
        self.end_node = None
        self.start_point = None
        self.end_point = None
        self.route = None
        self.annotated_texts = []
        self.current_zoom_level = 1.0
        self.text_visible = False
        self.calculated = False
        self.setting_start = True
        
        print("   -> Rendering map for interaction...")
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle(f"Penang Traffic Simulator - {time_label}", fontsize=14, fontweight='bold')
        
        plt.get_current_fig_manager().toolbar.zoom()
        plt.get_current_fig_manager().toolbar.pan()
        
        self.plot_base_map()
        self.add_control_buttons()
        
        self.cid_right_click = self.fig.canvas.mpl_connect('button_press_event', self.on_right_click)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_resize = self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        
        self.status_text = self.ax.text(0.02, 0.95, 
            "Ready to set START point (Right click)",
            transform=self.ax.transAxes, fontsize=10, color='green',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.instructions = self.ax.text(0.02, 0.98, 
            "Instructions:\n• RIGHT click (1st): Set START point\n• RIGHT click (2nd): Set END point\n• LEFT click + drag: Pan map\n• Scroll: Zoom in/out\n• 'Show Names': Toggle street names",
            transform=self.ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.update_title()
        plt.tight_layout()
        plt.show()

    def plot_base_map(self):
        self.ax.clear()
        edge_colors = []
        edge_widths = []
        edge_alpha = []
        
        for u, v, d in self.G.edges(data=True):
            if d.get('is_obstacle') == 1:
                edge_colors.append('#ff0000')
                edge_widths.append(3.0)
                edge_alpha.append(0.8)
            else:
                hw_type = str(clean_attribute(d.get('highway'), 'unclassified'))
                if 'primary' in hw_type:
                    edge_colors.append('#333333')
                    edge_widths.append(2.0)
                elif 'secondary' in hw_type:
                    edge_colors.append('#666666')
                    edge_widths.append(1.5)
                else:
                    edge_colors.append('#999999')
                    edge_widths.append(1.0)
                edge_alpha.append(0.6)
        
        ox.plot_graph(self.G, ax=self.ax, 
                     edge_color=edge_colors,
                     edge_linewidth=edge_widths,
                     edge_alpha=edge_alpha,
                     node_size=0,
                     show=False,
                     bgcolor='white')
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        
        if self.start_point:
            self.ax.scatter(self.start_point[0], self.start_point[1], 
                          c='green', s=200, marker='o', label='Start', 
                          zorder=10, edgecolors='black', linewidth=2)
        
        if self.end_point:
            self.ax.scatter(self.end_point[0], self.end_point[1], 
                          c='red', s=200, marker='s', label='End', 
                          zorder=10, edgecolors='black', linewidth=2)
        
        if self.route:
            ox.plot_graph_route(self.G, self.route, 
                               route_color='blue', 
                               route_linewidth=4,
                               ax=self.ax,
                               show=False)
        
        self.ax.legend(loc='upper right')
        self.draw_annotations()

    def add_control_buttons(self):
        button_ax = plt.axes([0.02, 0.02, 0.15, 0.05])
        names_ax = plt.axes([0.18, 0.02, 0.15, 0.05])
        clear_ax = plt.axes([0.34, 0.02, 0.15, 0.05])
        new_route_ax = plt.axes([0.50, 0.02, 0.15, 0.05])
        
        self.calc_button = Button(button_ax, 'Calculate Route', color='lightblue')
        self.calc_button.on_clicked(self.on_calculate_click)
        
        self.names_button = Button(names_ax, 'Show Names', color='lightgreen')
        self.names_button.on_clicked(self.on_toggle_names)
        
        self.clear_button = Button(clear_ax, 'Clear Route', color='lightcoral')
        self.clear_button.on_clicked(self.on_clear_click)
        
        self.new_route_button = Button(new_route_ax, 'New Route', color='lightyellow')
        self.new_route_button.on_clicked(self.on_new_route_click)

    def draw_annotations(self):
        for text in self.annotated_texts:
            text.remove()
        self.annotated_texts.clear()
        
        if not self.text_visible:
            return
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        view_width = xlim[1] - xlim[0]
        
        if view_width < 0.002:
            font_size = 10
            show_all = True
        elif view_width < 0.005:
            font_size = 9
            show_all = True
        elif view_width < 0.01:
            font_size = 8
            show_all = True
        elif view_width < 0.02:
            font_size = 7
            show_all = False
        else:
            return
        
        view_path = Path([(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1])])
        added_names = set()
        
        for u, v, d in self.G.edges(data=True):
            u_node = self.G.nodes[u]
            v_node = self.G.nodes[v]
            mid_x = (u_node['x'] + v_node['x']) / 2
            mid_y = (u_node['y'] + v_node['y']) / 2
            
            if not view_path.contains_point((mid_x, mid_y)):
                continue
            
            name = clean_attribute(d.get('name'), '')
            if not name or len(name) == 0:
                continue
            
            hw_type = str(clean_attribute(d.get('highway'), 'unclassified'))
            if not show_all:
                if hw_type not in ['primary', 'primary_link', 'secondary', 'secondary_link']:
                    continue
            
            name_key = f"{name[:20]}_{int(mid_x*1000)}_{int(mid_y*1000)}"
            if name_key in added_names:
                continue
            added_names.add(name_key)
            
            try:
                dx = v_node['x'] - u_node['x']
                dy = v_node['y'] - u_node['y']
                angle = math.degrees(math.atan2(dy, dx))
                if angle > 90 or angle < -90:
                    angle += 180
                
                text = self.ax.text(mid_x, mid_y, name, 
                                   fontsize=font_size, 
                                   alpha=0.7,
                                   ha='center', 
                                   va='center',
                                   rotation=angle,
                                   rotation_mode='anchor',
                                   bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', 
                                            edgecolor='none',
                                            alpha=0.7))
                self.annotated_texts.append(text)
            except Exception:
                continue

    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        zoom_factor = 1.2 if event.button == 'up' else 1/1.2
        self.current_zoom_level *= zoom_factor
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) / zoom_factor
        y_range = (ylim[1] - ylim[0]) / zoom_factor
        
        self.ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
        self.ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
        
        if self.text_visible: self.draw_annotations()
        self.fig.canvas.draw()

    def on_resize(self, event):
        if self.text_visible: self.draw_annotations()
        self.fig.canvas.draw()

    def on_right_click(self, event):
        if event.button != 3 or event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        
        if self.setting_start:
            self.start_node = ox.nearest_nodes(self.G, x, y)
            self.start_point = (x, y)
            self.setting_start = False
            self.status_text.set_text("Ready to set END point (Right click)")
            self.status_text.set_color('red')
        else:
            self.end_node = ox.nearest_nodes(self.G, x, y)
            self.end_point = (x, y)
            self.status_text.set_text("Ready to calculate route (Click 'Calculate Route')")
            self.status_text.set_color('blue')
        
        self.plot_base_map()
        self.update_title()
        self.fig.canvas.draw()

    def on_calculate_click(self, event):
        if not self.start_node or not self.end_node:
            print("⚠️ Please select both start and end points first!")
            return
        
        try:
            self.route = nx.shortest_path(self.G, self.start_node, self.end_node, weight='weight')
            self.calculated = True
            
            route_length = sum(self.G[u][v][0]['length'] for u, v in zip(self.route[:-1], self.route[1:]))
            route_time = sum(self.G[u][v][0].get('travel_time', 0) for u, v in zip(self.route[:-1], self.route[1:]))
            
            self.status_text.set_text(f"✓ Route found! Distance: {route_length:.0f}m, Time: {route_time:.0f}s")
            self.status_text.set_color('green')
            
            self.plot_base_map()
            self.update_title(route_length, route_time)
            
            img_path = os.path.join(TARGET_DIR, f"interactive_result_{self.time_label}.png")
            self.fig.savefig(img_path, dpi=300, bbox_inches='tight')
            print(f"💾 Route image saved to: {img_path}")
            
            self.calc_button.label.set_text('Recalculate')
            
        except nx.NetworkXNoPath:
            print("❌ No path found due to obstacles or connectivity issues.")
            self.ax.set_title("No path found! (Blocked by traffic)", fontsize=12, color='red')
            self.status_text.set_text("❌ No path found! Try different points")
            self.status_text.set_color('red')
        
        self.fig.canvas.draw()

    def on_toggle_names(self, event):
        self.text_visible = not self.text_visible
        self.names_button.label.set_text('Hide Names' if self.text_visible else 'Show Names')
        self.draw_annotations()
        self.fig.canvas.draw()

    def on_clear_click(self, event):
        self.route = None
        self.calculated = False
        if self.start_node and self.end_node:
            self.status_text.set_text("Ready to calculate route (Click 'Calculate Route')")
            self.status_text.set_color('blue')
        elif self.start_node:
            self.status_text.set_text("Ready to set END point (Right click)")
            self.status_text.set_color('red')
        else:
            self.status_text.set_text("Ready to set START point (Right click)")
            self.status_text.set_color('green')
        
        self.calc_button.label.set_text('Calculate Route')
        self.plot_base_map()
        self.update_title()
        self.fig.canvas.draw()

    def on_new_route_click(self, event):
        self.start_node = None
        self.end_node = None
        self.start_point = None
        self.end_point = None
        self.route = None
        self.calculated = False
        self.setting_start = True
        self.status_text.set_text("Ready to set START point (Right click)")
        self.status_text.set_color('green')
        self.calc_button.label.set_text('Calculate Route')
        self.plot_base_map()
        self.update_title()
        self.fig.canvas.draw()

    def update_title(self, route_length=None, route_time=None):
        if self.calculated and route_length:
            title = f"[{self.time_label}] ✓ Route Found! Distance: {route_length:.0f}m, Time: {route_time:.0f}s"
        elif self.start_node and self.end_node:
            title = f"[{self.time_label}] Start & End Selected - Click 'Calculate Route'"
        elif self.start_node:
            title = f"[{self.time_label}] Start Set - Right click again for End point"
        else:
            title = f"[{self.time_label}] Right click to select Start point"
        self.ax.set_title(title, fontsize=12, fontweight='bold')

# ==========================================
# 6. GUI Entry Point (No change)
# ==========================================
def start_launcher_gui(G_base):
    root = tk.Tk()
    root.title("AI Traffic Navigator (Penang)")
    root.geometry("400x300")
    
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
        root.destroy()
        G_sim = G_base.copy()
        G_sim = apply_traffic_conditions(G_sim, time_label)
        InteractiveRoutePlanner(G_sim, time_label)

    btn = ttk.Button(root, text="Open Interactive Map", command=on_launch)
    btn.pack(pady=20)
    
    tk.Label(root, text="Instructions:\n1. Select time period.\n2. Click button to open map.\n3. RIGHT click to set points.\n4. LEFT click to pan map.", 
             font=("Arial", 9), fg="gray", justify="center").pack(side=tk.BOTTOM, pady=20)

    root.mainloop()

def main():
    if not os.path.exists(TARGET_DIR):
        try: os.makedirs(TARGET_DIR)
        except: pass

    print(f"⬇️  Initializing Map Data for {TARGET_LOCATION}...")
    cf = '["highway"!~"motorway|trunk|motorway_link|trunk_link"]'
    G = ox.graph_from_point(TARGET_LOCATION, dist=SEARCH_RADIUS, custom_filter=cf, network_type='drive')
    
    G = ox.add_edge_bearings(G) 
    G = ox.add_edge_speeds(G, fallback=40) 
    
    for u, v, d in G.edges(data=True):
        if 'speed_kph' not in d or pd.isna(d['speed_kph']): d['speed_kph'] = 40.0
        if 'length' not in d or pd.isna(d['length']): d['length'] = 50.0

    G = ox.add_edge_travel_times(G)
    
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G_base = G.subgraph(largest_cc).copy()
    print("✅ Map Initialized!")

    # Generate CSV with NEW features
    csv_path = os.path.join(TARGET_DIR, "penang_traffic_data_english.csv")
    if not os.path.exists(csv_path):
        data = generate_training_data(G_base)
        df = pd.DataFrame(data)
        # Ensure 'Should_Take' is the last column (Weka prefers class at end)
        cols = [c for c in df.columns if c != 'Should_Take'] + ['Should_Take']
        df = df[cols]
        df.to_csv(csv_path, index=False)
        print(f"✅ Training data generated with {len(df)} rows: {csv_path}")
    else:
        print(f"✅ Training data found at {csv_path}, skipping generation.")

    print("🖥️  Starting GUI...")
    start_launcher_gui(G_base)

if __name__ == "__main__":
    main()