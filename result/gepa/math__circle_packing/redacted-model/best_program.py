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
        radii: np.array of shape (n) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    centers = np.zeros((n, 2))

    # Precision-optimized hexagonal grid: 5+6+5+5+5 = 26 circles
    # Key improvements:
    # - Reduced edge offset (0.08991 vs 0.08994) for increased boundary utilization
    # - Tighter horizontal spacing in row 1 (0.1708145 vs 0.1708149) for equalization
    # - Adjusted vertical spacing across all rows to minimize radius variance and maximize sum
    idx = 0
    
    # Row 0 (bottom): 5 circles - optimized edge offset (0.08991)
    for i in range(5):
        centers[idx] = [0.08991 + i * 0.205045, 0.08991]
        idx += 1
    
    # Row 1: 6 circles (offset) - tighter horizontal spacing and refined vertical position
    for i in range(6):
        centers[idx] = [0.07276275 + i * 0.1708145, 0.267348]
        idx += 1
    
    # Row 2: 5 circles - adjusted vertical position for minimal radius variance
    for i in range(5):
        centers[idx] = [0.08991 + i * 0.205045, 0.444803]
        idx += 1
    
    # Row 3: 5 circles - adjusted vertical position to balance top/bottom symmetry
    for i in range(5):
        centers[idx] = [0.08991 + i * 0.205045, 0.622657]
        idx += 1
    
    # Row 4 (top): 5 circles - exact mirror of row 0 for consistent edge usage
    for i in range(5):
        centers[idx] = [0.08991 + i * 0.205045, 1.0 - 0.08991]
        idx += 1

    # Numerical safety clipping without compromising optimal positioning
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)

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
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Optimized iterative constraint enforcement:
    # - 650 iterations for maximum convergence guarantee
    # - Four-way alternating processing order for optimal constraint propagation
    # - Ultra-tight numerical tolerance (1e-18) for strict non-overlap enforcement
    for iteration in range(650):
        updated = False
        # Build list of all pairs sorted by distance
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                pairs.append((dist, i, j))
        
        # Four-way alternating processing order for optimal constraint propagation
        mod = iteration % 4
        if mod == 0:
            pairs.sort(key=lambda x: x[0])  # Closest first (resolves local conflicts)
        elif mod == 1:
            pairs.sort(key=lambda x: -x[0])  # Farthest first (improves global balance)
        elif mod == 2:
            pairs.sort(key=lambda x: (x[1], x[2]))  # Index order (improves mixing)
        else:
            pairs.sort(key=lambda x: (x[2], x[1]))  # Reverse index order (additional mixing)
            
        for dist, i, j in pairs:
            # Enforce strict non-overlap with ultra-tight numerical tolerance
            if radii[i] + radii[j] > dist + 1e-18:
                # Equalize radii proportionally to maximize sum while respecting constraints
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
                updated = True
        if not updated:
            break

    # Final validation: ensure no non-physical radii
    radii = np.maximum(radii, 1e-12)

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