import matplotlib.pyplot as plt
import numpy as np
import os

def create_baseline_diagram(output_path):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    layers = np.arange(5)
    z_vals = layers * 20  # from geometry.py: z_trk_mm = 0, 20, 40, 60, 80

    # Detector limits based on geometry:
    # 512 rows * 27 um = 13.8 mm, 1024 cols * 29 um = 29.7 mm
    # Half width ~15 mm. Let's say y goes from -15 to +15.
    y_min, y_max = -15, 15

    # Helper to plot layers and structural limits
    def draw_layers(ax):
        for z in z_vals:
            # Draw sensor planes
            ax.plot([z, z], [y_min, y_max], color='darkgray', linewidth=4, zorder=0, solid_capstyle='butt')
        ax.set_xticks(z_vals)
        ax.set_xticklabels([f'Layer {i}' for i in layers])
        ax.set_yticks([])
        ax.set_xlim(-10, 90)
        ax.set_ylim(-20, 20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylabel("y (mm)")

    # Simulated track hits and noise
    # Track 1: straight track traversing layers
    trk1 = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    # Track 2: deviating track
    trk2 = np.array([5.0, 2.0, -1.0, -5.0, -8.0])
    
    noise = [
        [10.0, -12.0],        # L0
        [8.0, -8.0],          # L1
        [12.0, -13.0, 6.0],   # L2
        [14.0, 0.0],          # L3
        [-14.0, 2.0]          # L4
    ]

    all_hits = []
    for i in range(5):
        h = [trk1[i], trk2[i]] + noise[i]
        all_hits.append(np.array(h))

    # =========================================================================
    # Panel 1: Hits & Detector Structure
    # =========================================================================
    ax = axs[0]
    draw_layers(ax)
    ax.set_title("1. Pixel -> TRK Coord\n(Detector Structure)", fontsize=13, fontweight='bold')
    for i, h in enumerate(all_hits):
        ax.scatter([z_vals[i]]*len(h), h, color='black', s=45, zorder=5)
    
    # Draw max slope line (e.g. from bottom left to top right)
    # The max range in z is 80. The max structural limit in y is 30 (-15 to +15) -> slope ~ 30/80 = 0.375
    # Let's just draw the theoretical limit
    ax.plot([0, 80], [-15, 15], color='red', linestyle=':', linewidth=2, zorder=4)
    ax.annotate("Max valid track:\nslope_max", xy=(40, 0), xytext=(45, -5),
                arrowprops=dict(facecolor='black', arrowstyle='->'), color='red', fontsize=10, ha='left')

    ax.text(40, -28, 'Hits mapped to layer planes (0, 20, 40... mm)\nSlope limits based on detector geometry', ha='center', color='dimgray', fontsize=11)

    # =========================================================================
    # Panel 2: Edge Building (Slope Window)
    # =========================================================================
    ax = axs[1]
    draw_layers(ax)
    ax.set_title("2. Edge Building\n(Slope Window)", fontsize=13, fontweight='bold')
    
    # Highlight a single point to show the cone
    src_z = z_vals[0]
    src_y = trk1[0]
    slope_max_y = 0.2  # Match baseline config
    
    # Draw cone from the selected point to layer 1
    dz = z_vals[1] - z_vals[0]
    y_upper = src_y + slope_max_y * dz
    y_lower = src_y - slope_max_y * dz
    
    # Fill cone area
    ax.fill_between([src_z, z_vals[1]], 
                    [src_y, y_lower], 
                    [src_y, y_upper], 
                    color='khaki', alpha=0.5, zorder=1)
    ax.plot([src_z, z_vals[1]], [src_y, y_upper], color='orange', linestyle='--', linewidth=1)
    ax.plot([src_z, z_vals[1]], [src_y, y_lower], color='orange', linestyle='--', linewidth=1)

    ax.scatter(src_z, src_y, color='red', s=80, marker='*', zorder=10)
    ax.annotate(r'$sv_{max}$ cone', xy=(10, src_y), color='darkgoldenrod', fontsize=11, fontweight='bold', ha='center')

    for i, h in enumerate(all_hits):
        ax.scatter([z_vals[i]]*len(h), h, color='black', s=45, zorder=5)
    
    # Connect valid edges
    for i in range(4):
        for src in all_hits[i]:
            for dst in all_hits[i+1]:
                slope = (dst - src) / 20.0
                if abs(slope) <= 0.25:  # illustrate a bit looser for visual
                    ax.plot([z_vals[i], z_vals[i+1]], [src, dst], color='royalblue', alpha=0.3, zorder=2)
                    
    ax.text(40, -28, 'Connect adjacent hits within $\\Delta y / \\Delta z < \\mathrm{slope}_{max}$\n(Yellow cone shows search window)', ha='center', color='dimgray', fontsize=11)


    # =========================================================================
    # Panel 3: Chain Seeding (dslope_max extension)
    # =========================================================================
    ax = axs[2]
    draw_layers(ax)
    ax.set_title("3. Chain Seeding\n($dslope_{max}$ Extension)", fontsize=13, fontweight='bold')
    
    for i, h in enumerate(all_hits):
        ax.scatter([z_vals[i]]*len(h), h, color='black', s=45, zorder=5)

    # Triplet
    ax.plot(z_vals[:3], trk1[:3], color='orange', linewidth=8, alpha=0.4, solid_capstyle='round', label='Triplet')
    # Quad
    ax.plot(z_vals[:4], trk2[:4], color='green', linewidth=8, alpha=0.4, solid_capstyle='round', label='Quadruplet')
    # Quint
    ax.plot(z_vals, trk1, color='purple', linewidth=2, marker='o', markersize=6, label='Quintuplet')
    
    # Illustrate dslope cone branching out
    mid_z = z_vals[2]
    mid_y = trk1[2]
    prev_y = trk1[1]
    
    # Slope of the previous segment
    curr_slope = (mid_y - prev_y) / 20.0
    dslope_max = 0.05 # exaggerated for diagram
    
    # Cone for extending the chain
    dz2 = z_vals[3] - mid_z
    y_ext_upper = mid_y + (curr_slope + dslope_max) * dz2
    y_ext_lower = mid_y + (curr_slope - dslope_max) * dz2
    
    ax.fill_between([mid_z, z_vals[3]], 
                    [mid_y, y_ext_lower], 
                    [mid_y, y_ext_upper], 
                    color='mediumpurple', alpha=0.3, zorder=1)
    ax.annotate(r'$dslope_{max}$', xy=(50, mid_y + 1), color='indigo', fontsize=10)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.text(40, -28, 'Extend chains requiring consistent trajectory:\n$|slope_{i} - slope_{i-1}| < dslope_{max}$', ha='center', color='dimgray', fontsize=11)

    # =========================================================================
    # Panel 4: Fit & Score & Reject
    # =========================================================================
    ax = axs[3]
    draw_layers(ax)
    ax.set_title("4. Fit, Score & Shared-Hit Reject\n(Greedy Algorithm)", fontsize=13, fontweight='bold')
    for i, h in enumerate(all_hits):
        ax.scatter([z_vals[i]]*len(h), h, color='lightgray', s=45, zorder=1) 

    ax.plot(z_vals, trk1, color='blue', linewidth=2, label='Ch 1 (Low $\\chi^2$, Kept)')
    ax.scatter(z_vals, trk1, color='blue', s=50, zorder=5)
    
    ax.plot(z_vals[:4], trk2[:4], color='red', linestyle='--', linewidth=2, label='Ch 2 (Rej)')
    ax.scatter(z_vals[:4], trk2[:4], color='red', s=50, zorder=5)
    
    # A third track sharing a hit with trk1
    trk3 = np.array([-12, -7, 0, 7, 12])
    ax.plot(z_vals, trk3, color='crimson', linestyle=':', linewidth=2, label='Ch 3 (Shared Hit, Rej)')
    ax.scatter(z_vals, trk3, color='crimson', s=40, marker='x', zorder=6)

    shared_z = z_vals[2]
    shared_x = trk1[2]
    ax.scatter([shared_z], [shared_x], facecolors='none', edgecolors='gold', s=300, linewidth=2.5, zorder=10)
    ax.annotate('Shared Hit\n(trk1 claims it)', xy=(shared_z, shared_x), xytext=(shared_z - 15, shared_x + 8),
                arrowprops=dict(facecolor='black', arrowstyle='->'), color='black', fontsize=10, fontweight='bold')

    ax.legend(loc='lower left', fontsize=10)
    ax.text(40, -28, 'Global line fit $\\rightarrow$ rank by $\\chi^2$.\nGreedy removal of shared-hit tracks.', ha='center', color='dimgray', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) 
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Schematic diagram successfully saved to {output_path}")

if __name__ == "__main__":
    output_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_algorithm.png')
    create_baseline_diagram(output_png)
