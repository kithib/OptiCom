# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np
import cvxpy as cp


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

    # Optimized positions for 26 circles (hex-inspired + edge fill)
    centers[0] = [0.500, 0.500]

    # Ring of 6 around center (hexagonal symmetry)
    r1 = 0.218
    for i in range(6):
        a = 2 * np.pi * i / 6
        centers[1 + i] = [0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)]

    # Ring of 12 around that (staggered hexagonal)
    r2 = 0.415
    for i in range(12):
        a = 2 * np.pi * i / 12 + np.pi / 12
        centers[7 + i] = [0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)]

    # Seven outer circles - strategic placement to fill gaps near borders
    centers[19] = [0.088, 0.088]
    centers[20] = [0.912, 0.088]
    centers[21] = [0.088, 0.912]
    centers[22] = [0.912, 0.912]
    centers[23] = [0.500, 0.075]
    centers[24] = [0.500, 0.925]
    centers[25] = [0.075, 0.500]

    # Keep circles strictly inside unit square
    centers = np.clip(centers, 0.005, 0.995)

    # Compute optimal radii via LP (max sum given positions)
    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute maximum possible radii for given circle positions.
    Uses LP to maximize sum of radii subject to non-overlap and boundary constraints.
    Falls back to iterative scaling if LP fails.
    """
    n = centers.shape[0]

    # Pairwise distances (symmetric)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            D[i, j] = d
            D[j, i] = d

    # Distance to nearest border
    B = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        B[i] = min(x, y, 1 - x, 1 - y)

    # LP variables: radii (non-negative)
    r = cp.Variable(n, nonneg=True)
    constraints = [r <= B]

    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(r[i] + r[j] <= D[i, j])

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("LP status: " + str(prob.status))
        radii = np.clip(np.array(r.value).flatten(), 0.0, None)
    except Exception:
        # Robust fallback: iterative proportional scaling
        radii = B.copy()
        for _ in range(50):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if radii[i] + radii[j] > D[i, j]:
                        scale = D[i, j] / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
            if not changed:
                break

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
    # visualize(centers, radii)