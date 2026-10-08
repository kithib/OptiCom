# EVOLVE-BLOCK-START
import numpy as np


def _compute_min_max_ratio(points: np.ndarray) -> float:
    """Compute the min/max distance ratio for a set of points."""
    diffs = points[:, np.newaxis] - points[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=2)
    upper_tri = dists[np.triu_indices_from(dists, k=1)]
    dmin = np.min(upper_tri)
    dmax = np.max(upper_tri)
    return dmin / dmax


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    n = 14
    d = 3

    # Deterministic symmetric initial arrangement: cube vertices + face centers
    np.random.seed(42)
    
    # Cube vertices (8 points)
    vertices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
    
    # Face centers (6 points)
    faces = np.array([[0.5, 0.5, 0], [0.5, 0.5, 1],
                      [0.5, 0, 0.5], [0.5, 1, 0.5],
                      [0, 0.5, 0.5], [1, 0.5, 0.5]], dtype=float)
    
    # Combine for initial 14 points and center them
    points = np.vstack([vertices, faces])
    points -= np.mean(points, axis=0)
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    # Apply gradient-ascent style optimization
    learning_rate = 0.01
    iterations = 200

    for _ in range(iterations):
        ratio = _compute_min_max_ratio(points)

        # Compute pairwise distance vectors
        diffs = points[:, np.newaxis] - points[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2, keepdims=True)

        # Avoid division by zero on diagonal
        dists[np.diag_indices(n)] = 1.0

        # Push points away from each other, normalized by distance squared
        # This helps increase minimum distance while bounding maximum distance
        forces = diffs / (dists ** 2 + 1e-8)
        total_force = np.sum(forces, axis=1)

        points += learning_rate * total_force

        # Renormalize to keep points bounded and maintain scale consistency
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points


# EVOLVE-BLOCK-END