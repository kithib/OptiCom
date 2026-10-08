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

    # 1 central circle + 8 in a middle ring + 16 in an outer ring (adjusted for square geometry)
    # Center
    centers[0] = [0.5, 0.5]

    # Middle ring: 8 circles at radius ~0.22 from center, angle offset to align with gaps
    mid_r = 0.22
    for i in range(8):
        angle = (2 * np.pi * i / 8) + (np.pi / 8)
        centers[i + 1] = [0.5 + mid_r * np.cos(angle), 0.5 + mid_r * np.sin(angle)]

    # Outer ring: 16 circles placed to fill corner/edge regions of the square
    # Tuned positions: corner/edge circles pulled in slightly to maximize radius,
    # inter-circle spacing balanced for higher total sum
    outer_positions = [
        # Corners (4) - tuned for maximum border-limited radius
        (0.112, 0.112),
        (0.888, 0.112),
        (0.112, 0.888),
        (0.888, 0.888),
        # Edge midpoints (4) - tuned for maximum edge-limited radius
        (0.5, 0.092),
        (0.5, 0.908),
        (0.092, 0.5),
        (0.908, 0.5),
        # Between corners and edge midpoints (8) - balanced spacing with neighbors
        (0.302, 0.112),
        (0.698, 0.112),
        (0.302, 0.888),
        (0.698, 0.888),
        (0.112, 0.302),
        (0.112, 0.698),
        (0.888, 0.302),
        (0.888, 0.698),
    ]
    for i, (x, y) in enumerate(outer_positions):
        centers[i + 9] = [x, y]

    # Deterministic tiny nudge to avoid degenerate symmetries that hurt edge utilization
    centers += np.random.default_rng(42).normal(0, 0.00112, centers.shape)

    # Ensure centers stay within (0.015, 0.985) bounds to allow positive radii
    centers = np.clip(centers, 0.015, 0.985)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses iterative shrinking to find consistent radii, then final
    LP-style per-circle maximization to boost the total sum.

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

    # Iterative pairwise shrinking to find consistent, non-overlapping radii
    changed = True
    iterations = 0
    max_iter = 500
    while changed and iterations < max_iter:
        changed = False
        iterations += 1
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    continue
                current_sum = radii[i] + radii[j]
                if current_sum > dist + 1e-12:
                    # Shrink both circles proportionally to exactly satisfy constraint
                    scale = dist / current_sum
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True

    # Post-optimization: expand each circle individually (in random order) until
    # it hits either the border or another circle. This recovers slack left by
    # the symmetric shrinking and boosts the sum of radii.
    order = np.arange(n)
    # Deterministic shuffle so the result is reproducible
    np.random.default_rng(7).shuffle(order)
    for _ in range(8):
        for i in order:
            if radii[i] <= 0:
                continue
            # Max radius allowed by the square borders
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            # Max radius allowed by other circles: for every other circle j,
            # r_i <= d_ij - r_j
            for j in range(n):
                if j == i:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                candidate = dist - radii[j]
                if candidate < max_r:
                    max_r = candidate
            if max_r > radii[i]:
                radii[i] = max_r

    # Conservative final safety pass: ensure strict non-overlap after main loop
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                # Minimal safety margin to guarantee no overlap while preserving sum
                safe_scale = min(scale, 0.999995)
                radii[i] *= safe_scale
                radii[j] *= safe_scale

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
    # visualize(centers, radii)