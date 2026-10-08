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
    
    # Hexagonal-like grid pattern with offset rows for better packing
    # Use 5 rows with varying counts: 5, 6, 5, 5, 5 = 26 circles
    
    # Row 0 (bottom)
    centers[0] = [0.1, 0.1]
    centers[1] = [0.3, 0.1]
    centers[2] = [0.5, 0.1]
    centers[3] = [0.7, 0.1]
    centers[4] = [0.9, 0.1]
    
    # Row 1 (offset, 6 circles)
    row1_y = 0.1 + 0.175
    centers[5] = [0.11, row1_y]
    centers[6] = [0.27, row1_y]
    centers[7] = [0.43, row1_y]
    centers[8] = [0.59, row1_y]
    centers[9] = [0.75, row1_y]
    centers[10] = [0.91, row1_y]
    
    # Row 2
    row2_y = row1_y + 0.175
    centers[11] = [0.1, row2_y]
    centers[12] = [0.3, row2_y]
    centers[13] = [0.5, row2_y]
    centers[14] = [0.7, row2_y]
    centers[15] = [0.9, row2_y]
    
    # Row 3
    row3_y = row2_y + 0.175
    centers[16] = [0.11, row3_y]
    centers[17] = [0.29, row3_y]
    centers[18] = [0.47, row3_y]
    centers[19] = [0.65, row3_y]
    centers[20] = [0.85, row3_y]
    
    # Row 4 (top)
    row4_y = row3_y + 0.175
    centers[21] = [0.1, row4_y]
    centers[22] = [0.3, row4_y]
    centers[23] = [0.5, row4_y]
    centers[24] = [0.7, row4_y]
    centers[25] = [0.9, row4_y]

    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.001, 0.999)

    # Compute maximum valid radii for this configuration using improved algorithm
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses iterative optimization for better radius distribution.

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
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Compute pairwise distances
    dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis] - centers) ** 2, axis=2))
    
    # Multiple iterations to converge to better solution
    for iteration in range(100):
        max_violation = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = dist_matrix[i, j]
                current_sum = radii[i] + radii[j]
                if current_sum > dist + 1e-12:
                    violation = current_sum - dist
                    max_violation = max(max_violation, violation)
                    # Redistribute the space
                    scale = dist / current_sum
                    radii[i] *= scale
                    radii[j] *= scale
        
        if max_violation < 1e-12:
            break
    
    # Second pass: try to expand radii where possible
    for i in range(n):
        # Find the limiting factor
        max_possible = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
        for j in range(n):
            if i != j:
                dist = dist_matrix[i, j]
                available = dist - radii[j]
                if available < max_possible:
                    max_possible = available
        
        # Increase radius if possible
        if max_possible > radii[i] + 1e-12:
            radii[i] = max_possible

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