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
    n = 26
    centers = np.zeros((n, 2), dtype=np.float64)

    # Optimized placement: dense inner hexagonal-like cluster of 9 circles for large uniform radii,
    # outer 17 circles placed at edge/corner positions fine-tuned to balance
    # border-limited radius and pairwise spacing to maximize overall sum.
    centers[0] = [0.5000, 0.5000]

    # Inner ring (8 circles): symmetric square-grid offsets around center,
    # spacing chosen to maximize the minimum of border-limited radius and pairwise-limited radius.
    centers[1] = [0.2960, 0.2960]
    centers[2] = [0.2960, 0.5000]
    centers[3] = [0.2960, 0.7040]
    centers[4] = [0.5000, 0.2960]
    centers[5] = [0.5000, 0.7040]
    centers[6] = [0.7040, 0.2960]
    centers[7] = [0.7040, 0.5000]
    centers[8] = [0.7040, 0.7040]

    # Outer ring (17 circles): edge-aligned positions fine-tuned to maximize
    # border-limited radius while minimizing pairwise overlap constraints.
    centers[9]  = [0.0835, 0.0835]
    centers[10] = [0.0835, 0.2520]
    centers[11] = [0.0835, 0.4172]
    centers[12] = [0.0835, 0.5828]
    centers[13] = [0.0835, 0.7480]
    centers[14] = [0.0835, 0.9165]

    centers[15] = [0.2520, 0.0835]
    centers[16] = [0.4172, 0.0835]
    centers[17] = [0.5828, 0.0835]
    centers[18] = [0.7480, 0.0835]
    centers[19] = [0.9165, 0.0835]

    centers[20] = [0.2520, 0.9165]
    centers[21] = [0.4172, 0.9165]
    centers[22] = [0.5828, 0.9165]
    centers[23] = [0.7480, 0.9165]
    centers[24] = [0.9165, 0.2520]
    centers[25] = [0.9165, 0.7480]

    # Strict containment inside unit square (avoid numerical boundary touches)
    centers = np.clip(centers, 1e-4, 1.0 - 1e-4)

    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Iterative proportional scaling ensures pairwise non-overlap constraints
    are fully resolved across all pairs (adjusting radii for one pair may
    affect others, hence repeated passes until convergence).

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n, dtype=np.float64)

    # Upper bound from container boundaries
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)

    # Iteratively enforce pairwise non-overlap constraints until stable
    tol = 1e-12
    for _ in range(200):
        max_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq <= 0.0:
                    # Degenerate co-located circles: collapse radius to 0
                    radii[i] = 0.0
                    radii[j] = 0.0
                    continue
                dist = np.sqrt(dist_sq)
                current_sum = radii[i] + radii[j]
                if current_sum > dist:
                    violation = current_sum - dist
                    if violation > max_violation:
                        max_violation = violation
                    scale = dist / current_sum
                    radii[i] *= scale
                    radii[j] *= scale
        if max_violation < tol:
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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

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
    visualize(centers, radii)