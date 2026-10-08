# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import minimize


def construct_packing_base():
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

    # Recombined strategy:
    # 1. Hexagonal grid pattern (highest density) from exemplars
    # 2. Edge-aligned circles (no arbitrary clipping)
    # 3. Optimized row counts: 5, 5, 5, 5, 6 = 26 circles
    # 4. Tuned hexagonal spacing from exemplar 2 (tighter packing)
    # 5. Fine-tuned individual positions for maximum space utilization
    
    sqrt3 = np.sqrt(3)
    row_spacing = sqrt3 / 9.3  # Optimized vertical spacing - tighter for better density
    
    idx = 0
    # Row 0 (bottom): 5 circles - carefully placed x positions
    centers[idx] = [0.108, 0.108]; idx += 1
    centers[idx] = [0.304, 0.106]; idx += 1
    centers[idx] = [0.500, 0.108]; idx += 1
    centers[idx] = [0.696, 0.106]; idx += 1
    centers[idx] = [0.892, 0.108]; idx += 1
    
    # Row 1 (offset): 5 circles - proper hexagonal offset
    centers[idx] = [0.162, 0.108 + row_spacing]; idx += 1
    centers[idx] = [0.356, 0.106 + row_spacing]; idx += 1
    centers[idx] = [0.550, 0.108 + row_spacing]; idx += 1
    centers[idx] = [0.744, 0.106 + row_spacing]; idx += 1
    centers[idx] = [0.878, 0.108 + row_spacing]; idx += 1
    
    # Row 2: 5 circles - middle row
    centers[idx] = [0.108, 0.108 + 2 * row_spacing]; idx += 1
    centers[idx] = [0.304, 0.106 + 2 * row_spacing]; idx += 1
    centers[idx] = [0.500, 0.108 + 2 * row_spacing]; idx += 1
    centers[idx] = [0.696, 0.106 + 2 * row_spacing]; idx += 1
    centers[idx] = [0.892, 0.108 + 2 * row_spacing]; idx += 1
    
    # Row 3 (offset): 5 circles - upper offset row
    centers[idx] = [0.162, 0.108 + 3 * row_spacing]; idx += 1
    centers[idx] = [0.356, 0.106 + 3 * row_spacing]; idx += 1
    centers[idx] = [0.550, 0.108 + 3 * row_spacing]; idx += 1
    centers[idx] = [0.744, 0.106 + 3 * row_spacing]; idx += 1
    centers[idx] = [0.878, 0.108 + 3 * row_spacing]; idx += 1
    
    # Row 4 (top): 6 circles - optimally spaced to fill remaining space
    centers[idx] = [0.086, 0.892]; idx += 1
    centers[idx] = [0.240, 0.894]; idx += 1
    centers[idx] = [0.400, 0.892]; idx += 1
    centers[idx] = [0.600, 0.894]; idx += 1
    centers[idx] = [0.760, 0.892]; idx += 1
    centers[idx] = [0.914, 0.894]; idx += 1

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def construct_packing():
    """
    Optimized constructor: numerical optimization of circle positions and radii
    to maximize sum of radii.
    """
    centers0, _, _ = construct_packing_base()
    n = centers0.shape[0]
    
    def objective(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        return -np.sum(radii)  # Maximize sum of radii
    
    def constraints():
        cons = []
        # Distance to left/right/bottom/top walls
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*n:][i] - 0.001})  # r >= 0.001
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i:2*i+2][0] - x[2*n:][i]})  # x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i:2*i+2][0] - x[2*n:][i]})  # 1-x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i:2*i+2][1] - x[2*n:][i]})  # y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i:2*i+2][1] - x[2*n:][i]})  # 1-y >= r
        # Pairwise distance constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2) - (x[2*n+i] + x[2*n+j])
                })
        return cons
    
    r0 = np.array([min(min(c[0], c[1]), min(1-c[0], 1-c[1])) for c in centers0]) * 0.5
    x0 = np.concatenate([centers0.flatten(), r0])
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints(),
        bounds=[(0, 1)] * (2*n) + [(0.001, 0.5)] * n,
        options={'maxiter': 500, 'ftol': 1e-8}
    )
    
    centers_opt = result.x[:2*n].reshape(n, 2)
    radii_opt = result.x[2*n:]
    return centers_opt, radii_opt, np.sum(radii_opt)

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
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Use a more robust approach: iteratively reduce radii based on constraints
    # This ensures all constraints are properly satisfied at convergence
    for iteration in range(50):
        updated = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist + 1e-10:
                    excess = radii[i] + radii[j] - dist
                    # Reduce both radii proportionally
                    total = radii[i] + radii[j]
                    if total > 0:
                        radii[i] -= excess * (radii[i] / total) * 0.5
                        radii[j] -= excess * (radii[j] / total) * 0.5
                        updated = True
        if not updated:
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
    visualize(centers, radii)