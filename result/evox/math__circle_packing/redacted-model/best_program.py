# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Optimized hexagonal grid pattern (5-6-5-5-5) with ultra-high precision coordinates
    # Symmetric layout carefully balanced to minimize wasted space
    # Hexagonal packing vertical spacing: ~0.1732 (theoretical: sqrt(3)/2 * dx)
    # Fine-tuned all coordinates for optimal boundary clearance and inter-circle spacing
    
    # Row 0 (bottom) - 5 circles, perfectly symmetric, tight boundary clearance
    centers[0] = [0.0968, 0.0968]
    centers[1] = [0.2984, 0.0968]
    centers[2] = [0.5000, 0.0968]
    centers[3] = [0.7016, 0.0968]
    centers[4] = [0.9032, 0.0968]
    
    # Row 1 (staggered) - 6 circles, fills the width better with slight stagger adjustment
    centers[5] = [0.0812, 0.2700]
    centers[6] = [0.2478, 0.2700]
    centers[7] = [0.4167, 0.2700]
    centers[8] = [0.5856, 0.2700]
    centers[9] = [0.7522, 0.2700]
    centers[10] = [0.9188, 0.2700]
    
    # Row 2 - 5 circles, middle horizontal, very slightly compressed for row spacing
    centers[11] = [0.0966, 0.4430]
    centers[12] = [0.2983, 0.4430]
    centers[13] = [0.5000, 0.4430]
    centers[14] = [0.7017, 0.4430]
    centers[15] = [0.9034, 0.4430]
    
    # Row 3 (staggered) - 5 circles, optimized asymmetry for better packing density
    centers[16] = [0.1142, 0.6158]
    centers[17] = [0.3148, 0.6158]
    centers[18] = [0.5000, 0.6158]
    centers[19] = [0.6852, 0.6158]
    centers[20] = [0.8858, 0.6158]
    
    # Row 4 (top) - 5 circles, top portion with tight boundary packing
    centers[21] = [0.0980, 0.9020]
    centers[22] = [0.2990, 0.9020]
    centers[23] = [0.5000, 0.9020]
    centers[24] = [0.7010, 0.9020]
    centers[25] = [0.9020, 0.9020]

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively converge on valid radii - many passes with ultra-tight tolerance
    for _ in range(100):
        max_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                
                # If current radii would cause overlap
                if radii[i] + radii[j] > dist + 1e-16:
                    # Scale both radii proportionally to avoid overlap
                    violation = radii[i] + radii[j] - dist
                    max_violation = max(max_violation, violation)
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
        
        # Early exit if converged
        if max_violation < 1e-14:
            break
    
    # Additional refinement: see if we can increase radii without violating constraints
    # This gradient-ascent style step finds optimal individual radius increases
    for _ in range(10):
        improved = False
        for i in range(n):
            max_increase = np.inf
            # Check pairwise constraints
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                available = dist - radii[i] - radii[j]
                if available > 0:
                    max_increase = min(max_increase, available)
                else:
                    max_increase = 0
                    break
            
            # Check border constraint
            x, y = centers[i]
            border_available = min(x, y, 1 - x, 1 - y) - radii[i]
            max_increase = min(max_increase, border_available)
            
            if max_increase > 1e-12:
                radii[i] += max_increase * 0.5  # Conservative increase
                improved = True
    
    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)