import os
import math
import random
import warnings
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================
# Configuration
# ==========================
TARGET_DIR = r"D:\202509的课件\ai作业"
TARGET_LOCATION = (5.4195, 100.3325)  # George Town, Penang
SEARCH_RADIUS = 1200  # meters

# Animation settings (adjust if needed)
ANIMATION_INTERVAL_MS = 60  # milliseconds between frames
ANIMATION_FRAMES_PER_SEGMENT = 6  # how many frames to interpolate per segment
SAVE_ANIMATION = False  # Set True to save as MP4 (requires ffmpeg)
ANIMATION_FNAME = "route_animation.mp4"

# ==========================
# Helper utilities
# ==========================

def get_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from (lat1, lon1) to (lat2, lon2) in degrees."""
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    brng = math.atan2(y, x)
    brng = math.degrees(brng)
    return (brng + 360) % 360


def clean_attribute(val, default_val):
    """Handle lists, NaN, None gracefully."""
    if isinstance(val, list):
        return val[0] if len(val) > 0 else default_val
    if pd.isna(val) or val is None:
        return default_val
    return val


def get_edge_attr_safe(G, u, v):
    """Return an attribute dict for an edge (MultiDiGraph).
    If multiple parallel edges exist, return the first one.
    Returns None when no data available.
    """
    try:
        ed = G.get_edge_data(u, v)
        if ed is None:
            return None
        if isinstance(ed, dict):
            # choose first parallel edge attributes
            return list(ed.values())[0]
        return ed
    except Exception:
        return None


# ==========================
# Traffic simulation logic
# ==========================

def apply_traffic_conditions(G, time_label):
    """Mark some edges as congested obstacles based on time_label.
    This modifies G in-place and returns it.
    """
    all_edges = list(G.edges(keys=True, data=True))  # (u, v, key, data)

    if time_label == "Morning_Peak":
        congestion_rate = 0.20
        primary_weight = 90
    elif time_label == "Evening_Peak":
        congestion_rate = 0.15
        primary_weight = 70
    else:
        congestion_rate = 0.05
        primary_weight = 10

    weights = []
    for u, v, k, data in all_edges:
        hw_type = str(clean_attribute(data.get('highway'), 'unclassified'))
        if hw_type in ['primary', 'primary_link']:
            w = primary_weight
        elif hw_type in ['secondary', 'secondary_link']:
            w = int(primary_weight * 0.6)
        elif hw_type in ['tertiary']:
            w = 30
        else:
            w = 5
        weights.append(w)

    num_obstacles = int(len(all_edges) * congestion_rate)
    obstacle_keys = set()
    if num_obstacles > 0 and len(all_edges) > 0:
        # choose weighted indices
        selected_indices = random.choices(range(len(all_edges)), weights=weights, k=min(len(all_edges), num_obstacles * 3))
        # deduplicate preserving first order
        unique_selected = list(dict.fromkeys(selected_indices))
        chosen = unique_selected[:num_obstacles]
        for idx in chosen:
            u, v, k, data = all_edges[idx]
            obstacle_keys.add((u, v, k))

    # update attributes
    for u, v, k, data in all_edges:
        if (u, v, k) in obstacle_keys:
            G[u][v][k]['is_obstacle'] = 1
            G[u][v][k]['weight'] = 9999999
        else:
            G[u][v][k]['is_obstacle'] = 0
            G[u][v][k]['weight'] = G[u][v][k].get('travel_time', G[u][v][k].get('length', 1.0))

    return G


# ==========================
# Generate training data (optional)
# ==========================

def generate_training_data(G_base, trips_per_scenario=100):
    print("📊 Generating Weka training data...")
    all_data = []
    scenarios = ["Morning_Peak", "Off_Peak", "Evening_Peak"]

    for time_period in scenarios:
        G_scenario = G_base.copy()
        G_scenario = apply_traffic_conditions(G_scenario, time_period)
        nodes = list(G_scenario.nodes())
        print(f" -> scenario: {time_period} | nodes: {len(nodes)}")

        for i in range(trips_per_scenario):
            start, end = random.sample(nodes, 2)
            try:
                end_node_data = G_scenario.nodes[end]
                path = nx.shortest_path(G_scenario, start, end, weight='weight')
                path_edges = set(zip(path[:-1], path[1:]))

                def extract_row(u, v, d, label):
                    if d is None:
                        road_bearing = 0
                        maxspeed = 40.0
                        lanes = 1
                        length = 50.0
                        oneway = 0
                    else:
                        road_bearing = d.get('bearing', 0)
                        maxspeed = float(clean_attribute(d.get('speed_kph') or d.get('maxspeed'), 40))
                        lanes = int(clean_attribute(d.get('lanes'), 1))
                        length = float(d.get('length', 50.0))
                        oneway = 1 if d.get('oneway') else 0

                    u_node = G_scenario.nodes[u]
                    target_bearing = get_bearing(u_node['y'], u_node['x'], end_node_data['y'], end_node_data['x'])
                    angle_diff = abs(road_bearing - target_bearing)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    return {
                        "Time_Period": time_period,
                        "Road_Type": str(clean_attribute(d.get('highway') if d else None, 'unclassified')),
                        "Max_Speed": float(maxspeed),
                        "Lanes": int(lanes),
                        "Road_Length": float(length),
                        "Angle_Deviation": float(f"{angle_diff:.2f}"),
                        "One_Way": str(oneway),
                        "Is_Obstacle": int(d.get('is_obstacle', 0) if d else 0),
                        "Should_Take": label
                    }

                for u, v in path_edges:
                    d = get_edge_attr_safe(G_scenario, u, v)
                    all_data.append(extract_row(u, v, d, "YES"))

                available_edges = list(G_scenario.edges())
                sample_neg = random.sample(available_edges, min(4, len(available_edges)))
                for u, v in sample_neg:
                    if (u, v) not in path_edges:
                        d = get_edge_attr_safe(G_scenario, u, v)
                        all_data.append(extract_row(u, v, d, "NO"))

            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                print(f"warning: {e}")
                continue

    return all_data


# ==========================
# Interactive Route Planner with smooth animation
# ==========================
class InteractiveRoutePlanner:
    def __init__(self, G, time_label, animate=True):
        self.G = G
        self.time_label = time_label
        self.start_node = None
        self.end_node = None
        self.animate = animate

        print("Rendering interactive map...")
        edges_k_data = list(G.edges(keys=True, data=True))
        ec = ['#ff0000' if d.get('is_obstacle') == 1 else '#999999' for u, v, k, d in edges_k_data]
        ew = [2.0 if d.get('is_obstacle') == 1 else 0.6 for u, v, k, d in edges_k_data]

        self.fig, self.ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=ew, node_size=0, show=False, bgcolor='white')

        try:
            if hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager:
                self.fig.canvas.manager.set_window_title(f"Penang Traffic Simulator - {time_label}")
        except Exception:
            pass

        self.ax.set_title(f"[{time_label}] Click map to select START point", fontsize=12, color='green')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if self.start_node is None:
            self.start_node = ox.nearest_nodes(self.G, event.xdata, event.ydata)
            self.ax.scatter(event.xdata, event.ydata, c='green', s=90, zorder=10, edgecolors='black')
            self.ax.set_title(f"[{self.time_label}] Start set! Now click to select END point", fontsize=12, color='red')
            self.fig.canvas.draw()
            print(f"Start node: {self.start_node}")
            return

        if self.end_node is None:
            self.end_node = ox.nearest_nodes(self.G, event.xdata, event.ydata)
            if self.end_node == self.start_node:
                print("Start and End too close, choose another point.")
                return
            self.ax.scatter(event.xdata, event.ydata, c='red', s=90, zorder=10, edgecolors='black')
            self.ax.set_title(f"[{self.time_label}] Calculating route...", fontsize=12, color='blue')
            self.fig.canvas.draw()
            print(f"End node: {self.end_node}")
            self.calculate_and_animate()

    def calculate_and_animate(self):
        try:
            route = nx.shortest_path(self.G, self.start_node, self.end_node, weight='weight')
            print(f"Route nodes: {len(route)}")

            # coordinates (lon=x, lat=y) for route nodes
            coords = [(self.G.nodes[n]['x'], self.G.nodes[n]['y']) for n in route]

            # create interpolated frames along all segments
            frames = []
            for i in range(len(coords) - 1):
                x0, y0 = coords[i]
                x1, y1 = coords[i+1]
                for t in range(ANIMATION_FRAMES_PER_SEGMENT):
                    alpha = t / float(ANIMATION_FRAMES_PER_SEGMENT)
                    xi = x0 * (1 - alpha) + x1 * alpha
                    yi = y0 * (1 - alpha) + y1 * alpha
                    frames.append((xi, yi))
            # append final node
            frames.append(coords[-1])

            # Draw base route as thin gray line (static)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            self.ax.plot(xs, ys, linewidth=2, alpha=0.5, zorder=5)

            # create a moving marker (scatter)
            marker, = self.ax.plot([], [], marker='o', markersize=8, zorder=9, markeredgecolor='black', markerfacecolor='blue')

            # animation update function
            def update(frame_index):
                x, y = frames[frame_index]
                marker.set_data(x, y)
                return marker,

            ani = animation.FuncAnimation(self.fig, update, frames=len(frames), interval=ANIMATION_INTERVAL_MS, blit=True)

            # optionally save
            if SAVE_ANIMATION:
                try:
                    print("Saving animation (this may take a while)...")
                    ani.save(os.path.join(TARGET_DIR, ANIMATION_FNAME), writer='ffmpeg', dpi=200)
                    print(f"Animation saved: {os.path.join(TARGET_DIR, ANIMATION_FNAME)}")
                except Exception as e:
                    print(f"Failed to save animation: {e}")

            # save static image as well
            img_path = os.path.join(TARGET_DIR, f"interactive_result_{self.time_label}.png")
            try:
                self.fig.savefig(img_path, dpi=300, bbox_inches='tight')
                print(f"Saved static result: {img_path}")
            except Exception as e:
                print(f"Failed to save static image: {e}")

            # disconnect click handler to avoid extra routes
            self.fig.canvas.mpl_disconnect(self.cid)

            plt.show()

        except nx.NetworkXNoPath:
            self.ax.set_title("No path found (blocked)", color='red')
            self.fig.canvas.draw()
            print("No path found due to obstacles.")
        except Exception as e:
            print(f"Error when calculating route: {e}")


# ==========================
# GUI Launcher
# ==========================

def start_launcher_gui(G_base):
    root = tk.Tk()
    root.title("AI Traffic Navigator (Penang)")
    root.geometry("420x320")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 11), padding=8)

    ttk.Label(root, text="🚦 Penang Smart Traffic AI", font=("Arial", 16, "bold")).pack(pady=18)
    ttk.Label(root, text="Select Time Scenario:").pack(pady=6)

    time_var = tk.StringVar(value="Morning_Peak")
    time_combo = ttk.Combobox(root, textvariable=time_var, values=("Morning_Peak", "Evening_Peak", "Off_Peak"), state='readonly', width=32)
    time_combo.pack(pady=6)

    def on_launch():
        time_label = time_var.get()
        print(f"Launching map for: {time_label}")
        G_sim = G_base.copy()
        G_sim = apply_traffic_conditions(G_sim, time_label)
        root.destroy()
        InteractiveRoutePlanner(G_sim, time_label, animate=True)

    ttk.Button(root, text="Open Interactive Map", command=on_launch).pack(pady=14)

    ttk.Label(root, text="Instructions:\n1. Choose time period\n2. Click to open map\n3. Click map twice (start & end)", font=("Arial", 9), foreground='gray').pack(side=tk.BOTTOM, pady=12)

    root.mainloop()


# ==========================
# Main entry
# ==========================

def main():
    if not os.path.exists(TARGET_DIR):
        try:
            os.makedirs(TARGET_DIR)
        except Exception:
            pass

    print(f"Initializing map data around {TARGET_LOCATION}...")
    cf = '["highway"!~"motorway|trunk|motorway_link|trunk_link"]'
    G = ox.graph_from_point(TARGET_LOCATION, dist=SEARCH_RADIUS, custom_filter=cf, network_type='drive')

    # preprocess
    G = ox.add_edge_bearings(G)
    G = ox.add_edge_speeds(G, fallback=40)

    for u, v, k, d in G.edges(keys=True, data=True):
        if 'speed_kph' not in d or pd.isna(d.get('speed_kph')):
            G[u][v][k]['speed_kph'] = 40.0
        if 'length' not in d or pd.isna(d.get('length')):
            G[u][v][k]['length'] = 50.0

    G = ox.add_edge_travel_times(G)

    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G_base = G.subgraph(largest_cc).copy()
    print("Map initialized.")

    csv_path = os.path.join(TARGET_DIR, "penang_traffic_data_english.csv")
    if not os.path.exists(csv_path):
        data = generate_training_data(G_base)
        df = pd.DataFrame(data)
        try:
            df.to_csv(csv_path, index=False)
            print(f"Training CSV saved: {csv_path}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")
    else:
        print("Training CSV exists, skipping generation.")

    start_launcher_gui(G_base)


if __name__ == '__main__':
    main()
