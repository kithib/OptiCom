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
    centers = np.zeros((n, 2))

    # Design: 1 large center + 8 mid ring + 12 outer ring + 4 corners + 1 edge filler = 26.
    # Outer ring uses varying radius to better fill the square, following
    # cardinal/diagonal alignment trade-offs.

    # Large center circle
    centers[0] = [0.5, 0.5]

    # Ring of 8 circles around center (octagonal pattern, offset from axes to
    # create symmetric gaps beneficial for outer rings).
    r1 = 0.170
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8
        centers[i + 1] = [0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)]

    # Outer ring of 12 circles with varying radius: axis-aligned positions sit
    # closer to center; diagonal-aligned positions extend toward corners.
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        mod = i % 3
        if mod == 0:
            r = 0.336
        elif mod == 1:
            r = 0.384
        else:
            r = 0.390
        centers[i + 9] = [0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)]

    # Four corner fillers placed deep in each corner for corner-area capture.
    centers[21] = [0.120, 0.120]
    centers[22] = [0.880, 0.120]
    centers[23] = [0.120, 0.880]
    centers[24] = [0.880, 0.880]

    # One bottom-edge filler positioned to capture remaining bottom dead space.
    centers[25] = [0.5, 0.080]

    # Ensure circles are strictly within the unit square (tiny margin for safety)
    centers = np.clip(centers, 0.005, 0.995)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses iterative proportional scaling that converges to a valid solution.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    eps = 1e-9

    # Precompute pair-wise distances
    dists = np.sqrt(
        np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )

    # Distance from each circle center to the nearest square border
    border = np.min(
        np.column_stack(
            [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
        ),
        axis=1,
    )

    # Initialize radii at border limit; later reduced by pairwise constraints
    radii = border.copy()

    # Iteratively enforce pairwise constraints by scaling violating pairs.
    # Stable convergence: only shrink, never grow, monotonically decreasing.
    for _ in range(300):
        max_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d <= 0:
                    continue
                s = radii[i] + radii[j]
                if s > d + eps:
                    scale = d / s
                    radii[i] *= scale
                    radii[j] *= scale
                    if (s - d) > max_violation:
                        max_violation = s - d
        if max_violation < 1e-8:
            break

    # Final feasibility pass: take the minimum of per-circle constraints
    # derived from every other circle's current radius and border limits.
    for i in range(n):
        cap = border[i]
        for j in range(n):
            if i == j:
                continue
            cap = min(cap, dists[i, j] - radii[j])
        radii[i] = cap

    radii = np.clip(radii, 0.0, None)
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