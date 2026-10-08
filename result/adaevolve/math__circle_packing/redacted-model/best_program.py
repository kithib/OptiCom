"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Optimized hexagonal grid with 6 alternating rows
    # Hexagonal vertical spacing = sqrt(3)/2 * horizontal spacing
    # Reduced horizontal spacing to allow larger radii while maintaining hexagonal pattern
    h_spacing = 1.0 / 5.5  # Reduced spacing (tighter horizontal packing)
    v_spacing = h_spacing * np.sqrt(3) / 2.0  # Vertical spacing for hexagonal packing
    
    # Row 0 (bottom) - 6 circles, positions centered horizontally
    centers[0] = [1/12, 1/12]
    centers[1] = [1/12 + h_spacing, 1/12]
    centers[2] = [1/12 + 2*h_spacing, 1/12]
    centers[3] = [1/12 + 3*h_spacing, 1/12]
    centers[4] = [1/12 + 4*h_spacing, 1/12]
    centers[5] = [1/12 + 5*h_spacing, 1/12]
    
    # Row 1 (offset) - 5 circles
    centers[6] = [1/12 + h_spacing/2, 1/12 + v_spacing]
    centers[7] = [1/12 + 3*h_spacing/2, 1/12 + v_spacing]
    centers[8] = [1/12 + 5*h_spacing/2, 1/12 + v_spacing]
    centers[9] = [1/12 + 7*h_spacing/2, 1/12 + v_spacing]
    centers[10] = [1/12 + 9*h_spacing/2, 1/12 + v_spacing]
    
    # Row 2 - 6 circles
    centers[11] = [1/12, 1/12 + 2*v_spacing]
    centers[12] = [1/12 + h_spacing, 1/12 + 2*v_spacing]
    centers[13] = [1/12 + 2*h_spacing, 1/12 + 2*v_spacing]
    centers[14] = [1/12 + 3*h_spacing, 1/12 + 2*v_spacing]
    centers[15] = [1/12 + 4*h_spacing, 1/12 + 2*v_spacing]
    centers[16] = [1/12 + 5*h_spacing, 1/12 + 2*v_spacing]
    
    # Row 3 (offset) - 5 circles
    centers[17] = [1/12 + h_spacing/2, 1/12 + 3*v_spacing]
    centers[18] = [1/12 + 3*h_spacing/2, 1/12 + 3*v_spacing]
    centers[19] = [1/12 + 5*h_spacing/2, 1/12 + 3*v_spacing]
    centers[20] = [1/12 + 7*h_spacing/2, 1/12 + 3*v_spacing]
    centers[21] = [1/12 + 9*h_spacing/2, 1/12 + 3*v_spacing]
    
    # Row 4 - 4 circles positioned to maximize space usage in remaining vertical space
    centers[22] = [0.14, 1 - 1/11]
    centers[23] = [0.38, 1 - 1/11]
    centers[24] = [0.62, 1 - 1/11]
    centers[25] = [0.86, 1 - 1/11]
    
    # Clip to ensure everything is inside the unit square with small margin
    centers = np.clip(centers, 0.02, 0.98)
    
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
    
    # First, limit by distance to square borders
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Hybrid approach: first equalize, then use proportional scaling for convergence
    # Equalization helps distribute space more fairly initially
    for _ in range(5):  # Increased equalization iterations for better fairness
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                # If current radii would cause overlap
                if radii[i] + radii[j] > dist:
                    # Equalize: give each circle half the available distance
                    half_dist = dist / 2.0
                    if radii[i] > half_dist:
                        radii[i] = half_dist
                    if radii[j] > half_dist:
                        radii[j] = half_dist

    # Then use iterative proportional scaling to refine to valid solution
    # This ensures convergence while preserving relative radii sizes
    for iteration in range(200):  # Increased iterations for better convergence
        max_change = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                current_sum = radii[i] + radii[j]
                if current_sum > dist + 1e-15:
                    # Proportional scaling
                    scale = dist / current_sum
                    new_i = radii[i] * scale
                    new_j = radii[j] * scale
                    max_change = max(max_change, abs(radii[i] - new_i), abs(radii[j] - new_j))
                    radii[i] = new_i
                    radii[j] = new_j
        if max_change < 1e-14:  # Tighter convergence threshold
            break

    return radii


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