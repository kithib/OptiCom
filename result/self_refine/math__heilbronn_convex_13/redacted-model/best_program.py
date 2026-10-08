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
    
    # Optimized 3-fold symmetric pattern with 12 points on 2 concentric circles + center
    angles_outer = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points
    angles_inner = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6  # 6 rotated points
    
    # Fine-tuned radii ratio for optimal initial spacing
    outer_r = 0.492
    inner_r = 0.268
    
    outer_points = np.column_stack([0.5 + outer_r * np.cos(angles_outer), 
                                     0.5 + outer_r * np.sin(angles_outer)])
    inner_points = np.column_stack([0.5 + inner_r * np.cos(angles_inner), 
                                     0.5 + inner_r * np.sin(angles_inner)])
    center = np.array([[0.5, 0.5]])
    
    points = np.vstack([outer_points, inner_points, center])
    
    def normalize_hull_area(pts):
        """Normalize convex hull to unit area for consistent scoring."""
        try:
            hull = ConvexHull(pts)
            area = hull.volume  # For 2D, volume is the area
        except Exception:
            # Fallback: shoelace formula on angle-sorted points
            center_pt = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - center_pt[1], pts[:, 0] - center_pt[0])
            sorted_pts = pts[np.argsort(angles)]
            x, y = sorted_pts[:, 0], sorted_pts[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area < 1e-12:
            return pts
        
        scale = 1.0 / np.sqrt(area)
        center_pt = np.mean(pts, axis=0)
        return (pts - center_pt) * scale + center_pt

    points = normalize_hull_area(points)
    
    # Precompute triangle indices for efficiency
    tri_indices = list(combinations(range(n), 3))
    
    def compute_min_area(pts):
        pts_arr = np.asarray(pts)
        i_arr = np.array([t[0] for t in tri_indices])
        j_arr = np.array([t[1] for t in tri_indices])
        k_arr = np.array([t[2] for t in tri_indices])
        
        cross = (pts_arr[j_arr, 0] - pts_arr[i_arr, 0]) * (pts_arr[k_arr, 1] - pts_arr[i_arr, 1]) - \
                (pts_arr[j_arr, 1] - pts_arr[i_arr, 1]) * (pts_arr[k_arr, 0] - pts_arr[i_arr, 0])
        areas = 0.5 * np.abs(cross)
        return areas.min()
    
    best_points = points.copy()
    best_min = compute_min_area(best_points)
    
    # LogSumExp smoothed objective for better gradient guidance
    def compute_smoothed_objective(pts, alpha=200.0):
        pts_arr = np.asarray(pts)
        i_arr = np.array([t[0] for t in tri_indices])
        j_arr = np.array([t[1] for t in tri_indices])
        k_arr = np.array([t[2] for t in tri_indices])
        
        cross = (pts_arr[j_arr, 0] - pts_arr[i_arr, 0]) * (pts_arr[k_arr, 1] - pts_arr[i_arr, 1]) - \
                (pts_arr[j_arr, 1] - pts_arr[i_arr, 1]) * (pts_arr[k_arr, 0] - pts_arr[i_arr, 0])
        areas = 0.5 * np.abs(cross) + 1e-15  # Avoid log(0)
        
        # LogSumExp of negative areas approximates -min(area)
        weighted = np.exp(-alpha * areas)
        weights = weighted / weighted.sum()
        return -np.sum(weights * areas)
    
    # Gradient ascent on smoothed objective first
    for alpha in [50, 100, 200, 300, 400]:
        step = 0.002
        for epoch in range(70):
            obj_old = compute_smoothed_objective(best_points, alpha)
            grad = np.zeros_like(best_points)
            eps = 1e-6
            for i in range(n):
                for d in range(2):
                    pts_plus = best_points.copy()
                    pts_plus[i, d] += eps
                    pts_minus = best_points.copy()
                    pts_minus[i, d] -= eps
                    grad[i, d] = (compute_smoothed_objective(pts_plus, alpha) - 
                                  compute_smoothed_objective(pts_minus, alpha)) / (2 * eps)
            
            # Line search
            for lr in [step * 2, step, step * 0.5, step * 0.2]:
                new_pts = best_points + lr * grad
                new_pts = normalize_hull_area(new_pts)
                new_min = compute_min_area(new_pts)
                if new_min > best_min:
                    best_min = new_min
                    best_points = new_pts
                    break
    
    # Coordinate ascent with decreasing step sizes - increased directions and iterations
    for scale in [0.02, 0.018, 0.015, 0.012, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.0025, 0.002, 0.0015, 0.001]:
        for epoch in range(120):
            improved = False
            # Try moving each point with 24-directional search
            for idx in range(n):
                for ang in np.linspace(0, 2*np.pi, 24, endpoint=False):
                    dx, dy = scale * np.cos(ang), scale * np.sin(ang)
                    new_pts = best_points.copy()
                    new_pts[idx] += np.array([dx, dy])
                    new_pts = normalize_hull_area(new_pts)
                    new_min = compute_min_area(new_pts)
                    if new_min > best_min:
                        best_min = new_min
                        best_points = new_pts
                        improved = True
                        break
            if not improved:
                break
    
    # Perturbation-based escape from plateaus - more variations and hill climbing
    rng = np.random.default_rng(seed=123)
    for _ in range(20):
        for noise_scale in [0.025, 0.02, 0.015, 0.012, 0.008]:
            noisy_points = best_points + noise_scale * rng.standard_normal(best_points.shape)
            noisy_points = normalize_hull_area(noisy_points)
            # Quick hill climbing
            improved_escape = False
            for scale in [0.012, 0.008, 0.005, 0.003]:
                for epoch in range(50):
                    improved_epoch = False
                    for idx in range(n):
                        for ang in np.linspace(0, 2*np.pi, 16, endpoint=False):
                            dx, dy = scale * np.cos(ang), scale * np.sin(ang)
                            new_pts = noisy_points.copy()
                            new_pts[idx] += np.array([dx, dy])
                            new_pts = normalize_hull_area(new_pts)
                            new_min = compute_min_area(new_pts)
                            if new_min > best_min:
                                best_min = new_min
                                best_points = new_pts
                                noisy_points = new_pts.copy()
                                improved_epoch = True
                                improved_escape = True
                                break
                    if not improved_epoch:
                        break
            if improved_escape:
                break
    
    # Targeted refinement: expand smallest triangles by moving vertices away from centroid
    for _ in range(150):
        pts_arr = best_points
        i_arr = np.array([t[0] for t in tri_indices])
        j_arr = np.array([t[1] for t in tri_indices])
        k_arr = np.array([t[2] for t in tri_indices])
        
        cross = (pts_arr[j_arr, 0] - pts_arr[i_arr, 0]) * (pts_arr[k_arr, 1] - pts_arr[i_arr, 1]) - \
                (pts_arr[j_arr, 1] - pts_arr[i_arr, 1]) * (pts_arr[k_arr, 0] - pts_arr[i_arr, 0])
        areas = 0.5 * np.abs(cross)
        sorted_idx = np.argsort(areas)
        
        improved_any = False
        for tri_idx in sorted_idx[:20]:  # Focus on smallest 20 triangles
            i, j, k = tri_indices[tri_idx]
            area = areas[tri_idx]
            if area > best_min * 3.5:
                break  # Don't need to expand larger triangles
            
            centroid = (best_points[i] + best_points[j] + best_points[k]) / 3.0
            
            for pt_idx in [i, j, k]:
                vec = best_points[pt_idx] - centroid
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-10:
                    unit_vec = vec / vec_norm
                    # Also try perpendicular directions and 45-degree intermediates
                    perp_vec1 = np.array([-unit_vec[1], unit_vec[0]])
                    perp_vec2 = np.array([unit_vec[1], -unit_vec[0]])
                    diag_vec1 = (unit_vec + perp_vec1) / np.sqrt(2)
                    diag_vec2 = (unit_vec + perp_vec2) / np.sqrt(2)
                    
                    for dir_vec in [unit_vec, perp_vec1, perp_vec2, diag_vec1, diag_vec2]:
                        for step in [0.018, 0.015, 0.012, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002]:
                            new_pts = best_points.copy()
                            new_pts[pt_idx] += step * dir_vec
                            new_pts = normalize_hull_area(new_pts)
                            new_min = compute_min_area(new_pts)
                            if new_min > best_min:
                                best_min = new_min
                                best_points = new_pts
                                improved_any = True
                                break
                        if improved_any:
                            break
                    if improved_any:
                        break
            if improved_any:
                break
    
    return best_points