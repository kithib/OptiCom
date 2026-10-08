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

    # Hybrid pattern combining:
    # - Hexagonal close packing structure (from Exemplar 2)
    # - Optimized ring-inspired edge placement (from Exemplar 1)
    # Pattern: 5 rows with alternating offsets + corner optimization
    
    idx = 0
    
    # Row 1: 5 circles
    row_y = 0.15
    centers[idx] = [0.15, row_y]; idx += 1
    centers[idx] = [0.35, row_y]; idx += 1
    centers[idx] = [0.55, row_y]; idx += 1
    centers[idx] = [0.75, row_y]; idx += 1
    centers[idx] = [0.95, row_y]; idx += 1
    
    # Row 2: 6 circles (offset - hexagonal packing characteristic)
    row_y = 0.32
    centers[idx] = [0.10, row_y]; idx += 1
    centers[idx] = [0.28, row_y]; idx += 1
    centers[idx] = [0.46, row_y]; idx += 1
    centers[idx] = [0.64, row_y]; idx += 1
    centers[idx] = [0.82, row_y]; idx += 1
    centers[idx] = [0.98, row_y]; idx += 1
    
    # Row 3: 5 circles
    row_y = 0.50
    centers[idx] = [0.15, row_y]; idx += 1
    centers[idx] = [0.35, row_y]; idx += 1
    centers[idx] = [0.55, row_y]; idx += 1
    centers[idx] = [0.75, row_y]; idx += 1
    centers[idx] = [0.95, row_y]; idx += 1
    
    # Row 4: 6 circles (offset)
    row_y = 0.68
    centers[idx] = [0.10, row_y]; idx += 1
    centers[idx] = [0.28, row_y]; idx += 1
    centers[idx] = [0.46, row_y]; idx += 1
    centers[idx] = [0.64, row_y]; idx += 1
    centers[idx] = [0.82, row_y]; idx += 1
    centers[idx] = [0.98, row_y]; idx += 1
    
    # Row 5: 4 circles to reach total of 26
    row_y = 0.85
    centers[idx] = [0.20, row_y]; idx += 1
    centers[idx] = [0.45, row_y]; idx += 1
    centers[idx] = [0.70, row_y]; idx += 1
    centers[idx] = [0.95, row_y]; idx += 1

    # Apply fine adjustments - push corners outward for better border utilization
    # Inspired by Exemplar 2's corner optimization technique
    centers[0] = [0.12, 0.12]   # Bottom-left corner circle closer to corner
    centers[4] = [0.88, 0.12]   # Bottom-right corner circle closer to corner
    centers[22] = [0.12, 0.88]  # Top-left area circle adjustment
    centers[25] = [0.88, 0.88]  # Top-right corner circle closer to corner
    
    # Clip to ensure everything is inside the unit square with minimal border
    centers = np.clip(centers, 0.001, 0.999)

    # Compute maximum valid radii using enhanced iterative approach (from Exemplar 1)
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses iterative constrained optimization (from Exemplar 1) combined
    with validation passes (from Exemplar 2) for balanced, valid radii.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    
    # Initialize radii based on distance to borders - generous initial value
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iterative optimization - multiple passes for better convergence (Exemplar 1 technique)
    # Process circles multiple times since changing one radius affects others
    for iteration in range(50):
        max_change = 0.0
        for i in range(n):
            # Current maximum possible radius based on borders
            x, y = centers[i]
            border_limit = min(x, y, 1 - x, 1 - y)
            
            # Find minimum distance constraint from other circles
            min_available = border_limit
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    # Distance left after accounting for circle j's radius
                    available = dist - radii[j]
                    if available < min_available:
                        min_available = available
            
            # Ensure minimum positive radius (can't have radius <= 0)
            min_available = max(min_available, 1e-6)
            
            # Track maximum change for convergence detection
            change = abs(min_available - radii[i])
            if change > max_change:
                max_change = change
            
            radii[i] = min_available
        
        # Early stopping if changes become negligible
        if max_change < 1e-8:
            break
    
    # Final validation pass (Exemplar 2 technique) - guarantee no overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                adjustment = dist / (radii[i] + radii[j]) * 0.999
                radii[i] *= adjustment
                radii[j] *= adjustment
    
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