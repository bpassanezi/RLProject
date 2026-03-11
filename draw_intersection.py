import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_intersection(save_path='intersection_diagram.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    
    # Grid and background
    ax.set_facecolor('#F0F0F0')
    ax.grid(True, which='both', linestyle='--', color='lightgray', alpha=0.5)
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_yticks(np.arange(0, 301, 50))
    
    # 1. Road Structure
    road_width = 14.4  # Based on 2 lanes * ~7.2m (SUMO default/lane structure)
    half_width = road_width / 2
    
    # Horizontal Road
    ax.add_patch(patches.Rectangle((0, 150 - half_width), 142.8, road_width, color='gray', alpha=0.8, zorder=0))
    ax.add_patch(patches.Rectangle((157.2, 150 - half_width), 142.8, road_width, color='gray', alpha=0.8, zorder=0))
    
    # Vertical Road
    ax.add_patch(patches.Rectangle((150 - half_width, 0), road_width, 142.8, color='gray', alpha=0.8, zorder=0))
    ax.add_patch(patches.Rectangle((150 - half_width, 157.2), road_width, 142.8, color='gray', alpha=0.8, zorder=0))
    
    # Central Junction box
    ax.add_patch(patches.Rectangle((150 - half_width, 150 - half_width), road_width, road_width, color='darkgray', zorder=1))
    
    # Lane lines (center dashed)
    ax.plot([0, 142.8],color='white', linestyle='--', linewidth=2, zorder=2)
    ax.plot([157.2, 300], color='white', linestyle='--', linewidth=2, zorder=2)
    ax.plot([0, 142.8], color='white', linestyle='--', linewidth=2, zorder=2)
    ax.plot([157.2, 300], color='white', linestyle='--', linewidth=2, zorder=2)
    
    # 2. Add Node Labels
    nodes = {'t': (150, 150), 'n': (150, 290), 's': (150, 10), 'e': (290, 150), 'w': (10, 150)}
    for name, pos in nodes.items():
        circle = patches.Circle(pos, 5, color='black', ec='white', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(pos, pos, name.upper(), fontsize=16, fontweight='bold', color='white', ha='center', va='center', zorder=6)
        
    # Add Edge Labels
    ax.text(150, 216.4, 'n_t / t_n\n(142.8m)', ha='center', va='center', rotation=90)
    ax.text(150, 83.6, 's_t / t_s\n(142.8m)', ha='center', va='center', rotation=90)
    ax.text(223.6, 150, 'e_t / t_e (142.8m)', ha='center', va='center')
    ax.text(76.4, 150, 'w_t / t_w (142.8m)', ha='center', va='center')
    
    # 3. Permitted Flows (Arrows for one approach - North to simplify)
    arrow_props = dict(arrowstyle='-|>', mutation_scale=20, color='blue', linewidth=2.5, zorder=4)
    start_point = (150, 160) # Just inside junction from North
    
    # North straight to South
    ax.add_patch(patches.FancyArrowPatch(start_point, (150, 140), **arrow_props))
    # North right to West
    ax.add_patch(patches.FancyArrowPatch(start_point, (140, 150), connectionstyle="arc3,rad=.3", **arrow_props))
    # North left to East
    ax.add_patch(patches.FancyArrowPatch(start_point, (160, 150), connectionstyle="arc3,rad=-.3", **arrow_props))
    
    # Label flows
    ax.text(150, 170, 'Permitted:\nS, R, L', color='blue', fontweight='bold', ha='center')
    
    # 4. Prohibited Flow (U-Turn for North)
    u_arrow_props = dict(arrowstyle='-|>', mutation_scale=25, color='red', linewidth=3, zorder=4)
    
    # U-turn logic arc
    ax.add_patch(patches.FancyArrowPatch((147, 162), (153, 162), connectionstyle="arc3,rad=3.0", **u_arrow_props))
    
    # Prohibitory symbol (No U-Turn)
    # Prohibitory sign with explicit size relative to axes
    ax.add_patch(patches.Circle((150, 185), 7, color='red', fill=False, linewidth=4, zorder=10))
    # ax.plot(,, color='red', linewidth=4, zorder=11)
    ax.text(150, 178, 'U-Turns\nProhibited', color='red', fontsize=12, fontweight='bold', ha='center')
    
    plt.title('Thesis Intersection Schematic: Allowed Movements and Prohibitions', fontsize=18)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated diagram: {save_path}")
    plt.close()

if __name__ == "__main__":
    try:
        draw_intersection()
    except Exception as e:
        print(f"Error generating diagram: {e}")