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

    # Central large circle
    centers[0] = [0.5, 0.5]
    # First ring: 6 circles around center, hexagonal pattern
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers[i + 1] = [0.5 + 0.21 * np.cos(angle), 0.5 + 0.21 * np.sin(angle)]
    # Second ring: 12 circles, offset angles for denser packing
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        centers[i + 7] = [0.5 + 0.42 * np.cos(angle), 0.5 + 0.42 * np.sin(angle)]
    # Outer circles placed symmetrically near edges to fill corners and mid-sides
    # Corners (4)
    centers[19] = [0.070, 0.070]
    centers[20] = [0.930, 0.070]
    centers[21] = [0.070, 0.930]
    centers[22] = [0.930, 0.930]
    # Mid-sides (4): cover all four sides symmetrically to use available edge space
    centers[23] = [0.5, 0.070]
    centers[24] = [0.5, 0.930]
    centers[25] = [0.070, 0.5]

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))

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
    radii = np.ones(n, dtype=np.float64)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively resolve pairwise overlap constraints until stable
    for _ in range(500):
        updated = np.array(radii, copy=True)
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= 0:
                    # Degenerate case; shrink one circle to zero to avoid overlap
                    if updated[j] > 0:
                        updated[j] = 0
                        changed = True
                    continue
                if updated[i] + updated[j] > dist:
                    # Reduce the larger circle preferentially to preserve sum
                    if updated[i] >= updated[j]:
                        new_i = dist - updated[j]
                        if new_i < updated[i] and new_i >= 0:
                            updated[i] = new_i
                            changed = True
                    else:
                        new_j = dist - updated[i]
                        if new_j < updated[j] and new_j >= 0:
                            updated[j] = new_j
                            changed = True
        if not changed or np.allclose(radii, updated, atol=1e-12):
            break
        radii = updated

    # Final validation to ensure no overlaps and all circles are within bounds
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if radii[i] + radii[j] > dist + 1e-9:
                    valid = False
                    break
            if not valid:
                break
    if not valid:
        # Uniformly shrink to guarantee validity
        radii = radii * 0.999
        # Re-validate after shrink
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if radii[i] + radii[j] > dist:
                        valid = False
                        break
                if not valid:
                    break
        if not valid:
            radii = radii * 0.9

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