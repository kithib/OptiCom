import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    
    # Start with a hexagonal lattice arrangement that fills the unit square better
    points = []
    sqrt_n = 4
    hex_scale = 0.95  # Slightly expanded scaling to fill space
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            x = (i + 0.5 * (j % 2)) / (sqrt_n - 1) * hex_scale + (1 - hex_scale) / 2
            y = j / (sqrt_n - 1) * (np.sqrt(3) / 2) * hex_scale + (1 - hex_scale) / 2
            points.append([x, y])
    points = np.array(points[:n])
    
    # Add small perturbation - slightly reduced for better structure preservation
    points += rng.normal(0, 0.020, points.shape)
    points = np.clip(points, 0.025, 0.975)
    
    # Precompute triangle indices for efficiency
    tri_indices = list(combinations(range(n), 3))
    
    def compute_all_areas(pts):
        """Compute all triangle areas and return the list."""
        areas = np.empty(len(tri_indices))
        for idx, (i, j, k) in enumerate(tri_indices):
            v1x = pts[j, 0] - pts[i, 0]
            v1y = pts[j, 1] - pts[i, 1]
            v2x = pts[k, 0] - pts[i, 0]
            v2y = pts[k, 1] - pts[i, 1]
            areas[idx] = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        return areas
    
    def compute_min_area_smooth(pts, alpha=50.0):
        """Smooth approximation of min using LogSumExp."""
        areas = compute_all_areas(pts)
        # Numerically stable softmin with additional stability offset
        lse = -np.log(np.sum(np.exp(-alpha * areas + 1e-10)) + 1e-10) / alpha
        return lse, np.min(areas)
    
    def normalize_to_unit_area(pts):
        """Normalize points so their convex hull has unit area."""
        try:
            hull = ConvexHull(pts)
            area = hull.volume  # For 2D, ConvexHull.volume is the area
            if area > 0:
                scale = 1.0 / np.sqrt(area)
                pts = pts * scale
        except:
            pass
        return pts
    
    # Optimization setup - refined parameters for better convergence
    lr = 0.023  # Slightly higher initial learning rate
    momentum = 0.44
    velocity = np.zeros_like(points)
    best_points = points.copy()
    best_min_area = 0.0
    alpha = 50.0
    plateau_count = 0
    
    for iteration in range(850):  # More iterations for better refinement
        # Increase alpha over time for better min approximation
        alpha = min(50.0 + iteration * 0.17, 280.0)
        
        # Central difference gradient
        grad = np.zeros_like(points)
        eps = 1e-5
        for i in range(n):
            for d in range(2):
                points[i, d] += eps
                f_plus, _ = compute_min_area_smooth(points, alpha)
                points[i, d] -= 2 * eps
                f_minus, _ = compute_min_area_smooth(points, alpha)
                points[i, d] += eps
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient clipping and momentum update
        grad = np.clip(grad, -0.75, 0.75)
        velocity = momentum * velocity + lr * grad
        points += velocity
        points = np.clip(points, 0.005, 0.995)
        
        # Track best configuration based on actual min area
        areas = compute_all_areas(points)
        current_min = np.min(areas)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
            plateau_count = 0
        else:
            plateau_count += 1
        
        # Adaptive learning rate adjustments
        if (iteration + 1) % 85 == 0:
            lr *= 0.52
            momentum = min(momentum * 1.07, 0.80)
        
        # Reset on plateau but preserve best
        if plateau_count >= 50 and iteration < 650 and best_min_area > 0:
            points = best_points + rng.normal(0, 0.010, best_points.shape)
            points = np.clip(points, 0.01, 0.99)
            plateau_count = 0
            lr = max(lr * 0.75, 0.0025)
    
    # First refinement pass: improve small triangles - increased from 5 to 8 iterations
    for _ in range(8):
        final_areas = compute_all_areas(best_points)
        sorted_indices = np.argsort(final_areas)
        
        # Get top 20 smallest triangles (increased from 15) and try to improve
        for small_tri_idx in sorted_indices[:20]:
            i, j, k = tri_indices[small_tri_idx]
            for pt_idx in [i, j, k]:
                # Try more perturbation directions (increased from 8 to 12)
                for _ in range(12):
                    perturbation = rng.normal(0, 0.005, 2)
                    test_points = best_points.copy()
                    test_points[pt_idx] += perturbation
                    test_points = np.clip(test_points, 0.005, 0.995)
                    test_min = np.min(compute_all_areas(test_points))
                    if test_min > best_min_area:
                        best_points[pt_idx] = test_points[pt_idx]
                        best_min_area = test_min
    
    # Second refinement: increased from 3 to 5 iterations with smaller steps for finer control
    for _ in range(5):
        for i in range(n):
            best_dir = None
            best_improve = best_min_area
            for angle in np.linspace(0, 2 * np.pi, 16):  # Increased from 12 to 16 directions
                dx = 0.006 * np.cos(angle)  # Reduced step size from 0.008 to 0.006
                dy = 0.006 * np.sin(angle)
                test_points = best_points.copy()
                test_points[i, 0] += dx
                test_points[i, 1] += dy
                test_points = np.clip(test_points, 0.005, 0.995)
                test_min = np.min(compute_all_areas(test_points))
                if test_min > best_improve:
                    best_improve = test_min
                    best_dir = (dx, dy)
            if best_dir is not None:
                best_points[i, 0] += best_dir[0]
                best_points[i, 1] += best_dir[1]
                best_points = np.clip(best_points, 0.005, 0.995)
                best_min_area = best_improve
    
    # Third refinement pass: two-point perturbation to resolve degenerate cases
    for _ in range(3):
        final_areas = compute_all_areas(best_points)
        sorted_indices = np.argsort(final_areas)
        for small_tri_idx in sorted_indices[:10]:
            i, j, k = tri_indices[small_tri_idx]
            pts_in_tri = [i, j, k]
            for p1_idx in range(3):
                for p2_idx in range(p1_idx + 1, 3):
                    pt1 = pts_in_tri[p1_idx]
                    pt2 = pts_in_tri[p2_idx]
                    for _ in range(6):
                        pert1 = rng.normal(0, 0.004, 2)
                        pert2 = rng.normal(0, 0.004, 2)
                        test_points = best_points.copy()
                        test_points[pt1] += pert1
                        test_points[pt2] += pert2
                        test_points = np.clip(test_points, 0.005, 0.995)
                        test_min = np.min(compute_all_areas(test_points))
                        if test_min > best_min_area:
                            best_points[pt1] = test_points[pt1]
                            best_points[pt2] = test_points[pt2]
                            best_min_area = test_min
    
    # Normalize best configuration to unit area
    best_points = normalize_to_unit_area(best_points)
    
    return best_points