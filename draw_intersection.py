import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_intersection(save_path='intersection_diagram.png'):
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    
    # Grid and background styling
    ax.set_facecolor('#F0F0F0')
    ax.grid(True, which='both', linestyle='--', color='lightgray', alpha=0.5)
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_yticks(np.arange(0, 301, 50))
    
    # 1. Road Geometry
    road_width = 14.4  
    half_width = road_width / 2
    
    # Gray asphalt for Horizontal and Vertical roads
    ax.add_patch(patches.Rectangle((0, 150 - half_width), 142.8, road_width, color='gray', alpha=0.8, zorder=0))
    ax.add_patch(patches.Rectangle((157.2, 150 - half_width), 142.8, road_width, color='gray', alpha=0.8, zorder=0))
    ax.add_patch(patches.Rectangle((150 - half_width, 0), road_width, 142.8, color='gray', alpha=0.8, zorder=0))
    ax.add_patch(patches.Rectangle((150 - half_width, 157.2), road_width, 142.8, color='gray', alpha=0.8, zorder=0))
    
    # Central Junction Box (Node T)
    ax.add_patch(patches.Rectangle((150 - half_width, 150 - half_width), road_width, road_width, color='darkgray', zorder=1))
    
    # --- FIXED LANE LINES (Verified X and Y arrays) ---
    # Horizontal lines
    ax.plot([0, 142.8], [150, 150], color='white', linestyle='--', linewidth=2, zorder=2)
    ax.plot([157.2, 300], [150, 150], color='white', linestyle='--', linewidth=2, zorder=2)
    
    # Vertical lines
    ax.plot([150, 150], [0, 142.8], color='white', linestyle='--', linewidth=2, zorder=2)
    ax.plot([150, 150], [157.2, 300], color='white', linestyle='--', linewidth=2, zorder=2)
    
    # 2. Node Labels (N, S, E, W, T)
    nodes = {
        # 't': (150, 150),
    'n': (150, 290), 's': (150, 10), 'e': (290, 150), 'w': (10, 150)}
    for name, pos in nodes.items():
        circle = patches.Circle(pos, 8, color='black', ec='white', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], name.upper(), fontsize=14, fontweight='bold', color='white', ha='center', va='center', zorder=6)
        
    # Edge Labels
    ax.text(150, 220, 'n_t / t_n\n(142.8m)', ha='center', va='center', rotation=90, fontsize=10)
    ax.text(150, 80, 's_t / t_s\n(142.8m)', ha='center', va='center', rotation=90, fontsize=10)
    ax.text(225, 150, 'e_t / t_e (142.8m)', ha='center', va='center', fontsize=10)
    ax.text(75, 150, 'w_t / t_w (142.8m)', ha='center', va='center', fontsize=10)
    
    # 3. Permitted Flows (Blue Arrows)
    arrow_props = dict(arrowstyle='-|>', mutation_scale=20, color='blue', linewidth=2, zorder=4)
    start_pt = (150, 165)
    
    # Straight, Right, Left

    # ax.add_patch(patches.FancyArrowPatch(start_pt, (150, 135), **arrow_props))
    
    # ax.add_patch(patches.FancyArrowPatch(start_pt, (135, 150), connectionstyle="arc3,rad=.3", **arrow_props))
    # ax.add_patch(patches.FancyArrowPatch(start_pt, (165, 150), connectionstyle="arc3,rad=-.3", **arrow_props))
    
    # 3. Traffic Movements (The 12 Directions)
    # Format: (start_x, start_y, end_x, end_y, curvature_rad)
    movements = [
        # From North (Entering at 150, 165)
        (150, 165, 150, 135, 0),    # Straight to S
        (150, 165, 135, 150, 0.3),  # Right to W
        (150, 165, 165, 150, -0.3), # Left to E
        # From South (Entering at 150, 135)
        (150, 135, 150, 165, 0),    # Straight to N
        (150, 135, 165, 150, 0.3),  # Right to E
        (150, 135, 135, 150, -0.3), # Left to W
        # From West (Entering at 135, 150)
        (135, 150, 165, 150, 0),    # Straight to E
        (135, 150, 150, 165, 0.3),  # Right to N
        (135, 150, 150, 135, -0.3), # Left to S
        # From East (Entering at 165, 150)
        (165, 150, 135, 150, 0),    # Straight to W
        (165, 150, 150, 135, 0.3),  # Right to S
        (165, 150, 150, 165, -0.3), # Left to N
    ]

    for sx, sy, ex, ey, rad in movements:
        color = 'blue' if rad == 0 else 'green' if rad > 0 else 'orange'
        style = f"arc3,rad={rad}"
        arrow = patches.FancyArrowPatch((sx, sy), (ex, ey), connectionstyle=style,
                                        arrowstyle='-|>', mutation_scale=15, 
                                        color=color, linewidth=1.5, alpha=0.6, zorder=4)
        ax.add_patch(arrow)

    # Legend for the directions
    ax.text(10, 280, "■ Straight (N-S / E-W)", color='blue', fontweight='bold')
    ax.text(10, 270, "■ Right Turns", color='green', fontweight='bold')
    ax.text(10, 260, "■ Left Turns", color='orange', fontweight='bold')
    
    # # 4. Prohibited U-Turn (Red Slash and Arrow)
    # u_props = dict(arrowstyle='-|>', mutation_scale=20, color='red', linewidth=2, zorder=4)
    # ax.add_patch(patches.FancyArrowPatch((146, 165), (154, 165), connectionstyle="arc3,rad=3.5", **u_props))
    
    # The "No" Sign
    # ax.add_patch(patches.Circle((150, 195), 8, color='red', fill=False, linewidth=3, zorder=10))
    # ax.plot([144, 156], [201, 189], color='red', linewidth=3, zorder=11)
    
    ax.set_title('Thesis Intersection Geometry & Allowed Movements', fontsize=16, pad=20)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_intersection()