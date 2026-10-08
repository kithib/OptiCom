import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Fixed seed for determinism
    rng = np.random.default_rng(seed=42)
    
    # Evaluate multiple initialization strategies and select the best
    def init_9boundary_4interior():
        angles = np.linspace(0, 2 * np.pi, 10)[:9]
        angles += np.array([-0.01, 0.008, -0.006, 0.005, -0.004, 0.005, -0.006, 0.008, -0.01])
        radius = 0.485
        boundary_x = 0.5 + radius * np.cos(angles)
        boundary_y = 0.5 + radius * np.sin(angles)
        boundary_points = np.column_stack([boundary_x, boundary_y])
        interior_angles = np.array([0.125*np.pi, 0.625*np.pi, 1.125*np.pi, 1.625*np.pi])
        interior_r = 0.225
        interior_x = 0.5 + interior_r * np.cos(interior_angles)
        interior_y = 0.5 + interior_r * np.sin(interior_angles)
        interior_points = np.column_stack([interior_x, interior_y])
        pts = np.vstack([boundary_points[:5], interior_points, boundary_points[5:]])
        return pts + rng.normal(0, 0.0055, pts.shape)
    
    def init_10boundary_3interior():
        angles = np.linspace(0, 2 * np.pi, 11)[:10]
        radius = 0.488
        boundary_x = 0.5 + radius * np.cos(angles)
        boundary_y = 0.5 + radius * np.sin(angles)
        boundary_points = np.column_stack([boundary_x, boundary_y])
        interior_angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        interior_r = 0.20
        interior_x = 0.5 + interior_r * np.cos(interior_angles)
        interior_y = 0.5 + interior_r * np.sin(interior_angles)
        interior_points = np.column_stack([interior_x, interior_y])
        pts = np.vstack([boundary_points, interior_points])
        return pts + rng.normal(0, 0.005, pts.shape)
    
    def init_8boundary_5interior():
        angles = np.linspace(0, 2 * np.pi, 9)[:8]
        angles += np.array([-0.015, 0.012, -0.008, 0.015, -0.015, 0.008, -0.012, 0.015])
        radius = 0.482
        boundary_x = 0.5 + radius * np.cos(angles)
        boundary_y = 0.5 + radius * np.sin(angles)
        boundary_points = np.column_stack([boundary_x, boundary_y])
        interior_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4])
        interior_r = np.array([0.24, 0.24, 0.24, 0.24, 0.12])
        interior_x = 0.5 + interior_r * np.cos(interior_angles)
        interior_y = 0.5 + interior_r * np.sin(interior_angles)
        interior_points = np.column_stack([interior_x, interior_y])
        pts = np.vstack([boundary_points, interior_points])
        return pts + rng.normal(0, 0.0052, pts.shape)
    
    def init_11boundary_2interior():
        angles = np.linspace(0, 2 * np.pi, 12)[:11]
        angles += np.array([-0.008, 0.006, -0.004, 0.003, -0.002, 0.002, -0.003, 0.004, -0.006, 0.008, -0.005])
        radius = 0.49
        boundary_x = 0.5 + radius * np.cos(angles)
        boundary_y = 0.5 + radius * np.sin(angles)
        boundary_points = np.column_stack([boundary_x, boundary_y])
        interior_angles = np.array([0.25*np.pi, 1.25*np.pi])
        interior_r = 0.16
        interior_x = 0.5 + interior_r * np.cos(interior_angles)
        interior_y = 0.5 + interior_r * np.sin(interior_angles)
        interior_points = np.column_stack([interior_x, interior_y])
        pts = np.vstack([boundary_points, interior_points])
        return pts + rng.normal(0, 0.005, pts.shape)
    
    # Precompute all triangle triplets once for efficiency
    triplets = list(combinations(range(n), 3))
    triplets_arr = np.array(triplets)
    num_triplets = len(triplets)
    
    # Vectorized minimum triangle area computation - also returns areas array
    def compute_min_area(pts):
        a = pts[triplets_arr[:, 0]]
        b = pts[triplets_arr[:, 1]]
        c = pts[triplets_arr[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        return min_area
    
    # New: multi-criterion initialization evaluation - considers bottom-k triangles
    def compute_init_score(pts):
        a = pts[triplets_arr[:, 0]]
        b = pts[triplets_arr[:, 1]]
        c = pts[triplets_arr[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = 0.5 * np.abs(cross)
        k = 5
        smallest = np.partition(areas, k)[:k]
        return np.mean(smallest)
    
    # Evaluate initializations using improved scoring
    inits = [init_9boundary_4interior(), init_10boundary_3interior(), 
             init_8boundary_5interior(), init_11boundary_2interior()]
    init_scores = [compute_init_score(pts) for pts in inits]
    best_init_idx = np.argmax(init_scores)
    points = inits[best_init_idx]
    
    # Keep track of best configuration
    best_points = points.copy()
    best_min_area = compute_min_area(points)
    
    # Phase 1: Enhanced gradient ascent - annealed k and step sizes
    for iteration in range(45):
        a = points[triplets_arr[:, 0]]
        b = points[triplets_arr[:, 1]]
        c = points[triplets_arr[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = 0.5 * np.abs(cross)
        
        if iteration < 15:
            k_smallest = min(16, num_triplets - 1)
        elif iteration < 30:
            k_smallest = min(13, num_triplets - 1)
        else:
            k_smallest = min(10, num_triplets - 1)
        smallest_indices = np.argpartition(areas, k_smallest)[:k_smallest]
        
        for min_idx in smallest_indices:
            worst_tri = triplets_arr[min_idx]
            for i, pt_idx in enumerate(worst_tri):
                other_idxs = [j for j in worst_tri if j != pt_idx]
                midpoint = np.mean(points[other_idxs], axis=0)
                direction = points[pt_idx] - midpoint
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    if iteration < 15:
                        step = 0.018
                    elif iteration < 30:
                        step = 0.015
                    else:
                        step = 0.012
                    points[pt_idx] += step * direction / norm
        
        centroid = np.mean(points, axis=0)
        if iteration < 15:
            contraction = 0.967
        elif iteration < 30:
            contraction = 0.977
        else:
            contraction = 0.987
        points = contraction * (points - centroid) + centroid
        
        current_min = compute_min_area(points)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
    
    # Restore best from Phase 1
    points = best_points.copy()
    
    # Phase 2: Improved LogSumExp optimization
    lr = 0.038
    lse_scale = 115
    for iteration in range(40):
        a = points[triplets_arr[:, 0]]
        b = points[triplets_arr[:, 1]]
        c = points[triplets_arr[:, 2]]
        cross_vec = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = np.abs(cross_vec) / 2.0
        
        log_sum = np.sum(np.exp(-lse_scale * areas))
        grad = np.zeros_like(points)
        
        weights = np.exp(-lse_scale * areas) / (2.0 * log_sum)
        signs = np.where(np.abs(cross_vec) > 1e-12, cross_vec / np.abs(cross_vec), 0.0)
        
        for idx in range(num_triplets):
            i, j, k = triplets_arr[idx]
            s = signs[idx]
            if abs(s) > 0:
                w = weights[idx]
                ai, aj, ak = points[i], points[j], points[k]
                grad[i] += w * s * np.array([-(ak[1] - aj[1]), aj[0] - ak[0]]) / 2.0
                grad[j] += w * s * np.array([ai[1] - ak[1], ak[0] - ai[0]]) / 2.0
                grad[k] += w * s * np.array([aj[1] - ai[1], ai[0] - aj[0]]) / 2.0
        
        points = points - lr * grad
        lr = max(0.0085, lr * 0.93)
        
        current_min = compute_min_area(points)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
    
    # Phase 3: Fine refinement with higher LSE scale
    points = best_points.copy()
    lr = 0.027
    lse_scale = 150
    for iteration in range(30):
        a = points[triplets_arr[:, 0]]
        b = points[triplets_arr[:, 1]]
        c = points[triplets_arr[:, 2]]
        cross_vec = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = np.abs(cross_vec) / 2.0
        
        log_sum = np.sum(np.exp(-lse_scale * areas))
        grad = np.zeros_like(points)
        
        weights = np.exp(-lse_scale * areas) / (2.0 * log_sum)
        signs = np.where(np.abs(cross_vec) > 1e-12, cross_vec / np.abs(cross_vec), 0.0)
        
        for idx in range(num_triplets):
            i, j, k = triplets_arr[idx]
            w = weights[idx]
            s = signs[idx]
            if abs(s) > 0:
                grad[i] += w * s * np.array([-(points[k, 1] - points[j, 1]), points[j, 0] - points[k, 0]]) / 2.0
                grad[j] += w * s * np.array([points[i, 1] - points[k, 1], points[k, 0] - points[i, 0]]) / 2.0
                grad[k] += w * s * np.array([points[j, 1] - points[i, 1], points[i, 0] - points[j, 0]]) / 2.0
        
        points = points - lr * grad
        lr = max(0.0058, lr * 0.942)
        
        current_min = compute_min_area(points)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
    
    # Phase 4: Final ultra-fine tuning
    points = best_points.copy()
    lr = 0.014
    lse_scale = 175
    for iteration in range(20):
        a = points[triplets_arr[:, 0]]
        b = points[triplets_arr[:, 1]]
        c = points[triplets_arr[:, 2]]
        cross_vec = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = np.abs(cross_vec) / 2.0
        
        log_sum = np.sum(np.exp(-lse_scale * areas))
        grad = np.zeros_like(points)
        
        weights = np.exp(-lse_scale * areas) / (2.0 * log_sum)
        signs = np.where(np.abs(cross_vec) > 1e-12, cross_vec / np.abs(cross_vec), 0.0)
        
        for idx in range(num_triplets):
            i, j, k = triplets_arr[idx]
            s = signs[idx]
            if abs(s) > 0:
                w = weights[idx]
                grad[i] += w * s * np.array([-(points[k, 1] - points[j, 1]), points[j, 0] - points[k, 0]]) / 2.0
                grad[j] += w * s * np.array([points[i, 1] - points[k, 1], points[k, 0] - points[i, 0]]) / 2.0
                grad[k] += w * s * np.array([points[j, 1] - points[i, 1], points[i, 0] - points[j, 0]]) / 2.0
        
        points = points - lr * grad
        lr = max(0.0032, lr * 0.958)
        
        current_min = compute_min_area(points)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
    
    # Phase 5: Targeted improvement of bottom triangles
    points = best_points.copy()
    for iteration in range(25):
        a = points[triplets_arr[:, 0]]
        b = points[triplets_arr[:, 1]]
        c = points[triplets_arr[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Focus on bottom 8 triangles
        k_smallest = min(8, num_triplets - 1)
        smallest_indices = np.argpartition(areas, k_smallest)[:k_smallest]
        
        for min_idx in smallest_indices:
            worst_tri = triplets_arr[min_idx]
            for i, pt_idx in enumerate(worst_tri):
                other_idxs = [j for j in worst_tri if j != pt_idx]
                midpoint = np.mean(points[other_idxs], axis=0)
                direction = points[pt_idx] - midpoint
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    step = 0.011 - iteration * 0.00025
                    step = max(0.0045, step)
                    points[pt_idx] += step * direction / norm
        
        current_min = compute_min_area(points)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
    
    # Restoration sweep: Local escape with multiple seeds to avoid plateaus
    base_points = best_points.copy()
    base_min = best_min_area
    for esc_seed in [123, 456, 789, 246, 135]:
        esc_rng = np.random.default_rng(seed=esc_seed)
        points = base_points + esc_rng.normal(0, 0.004, base_points.shape)
        
        esc_lr = 0.012
        esc_lse_scale = 160
        for esc_iter in range(12):
            a = points[triplets_arr[:, 0]]
            b = points[triplets_arr[:, 1]]
            c = points[triplets_arr[:, 2]]
            cross_vec = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            areas = np.abs(cross_vec) / 2.0
            
            log_sum = np.sum(np.exp(-esc_lse_scale * areas))
            grad = np.zeros_like(points)
            
            weights = np.exp(-esc_lse_scale * areas) / (2.0 * log_sum)
            signs = np.where(np.abs(cross_vec) > 1e-12, cross_vec / np.abs(cross_vec), 0.0)
            
            for idx in range(num_triplets):
                i, j, k = triplets_arr[idx]
                s = signs[idx]
                if abs(s) > 0:
                    w = weights[idx]
                    grad[i] += w * s * np.array([-(points[k, 1] - points[j, 1]), points[j, 0] - points[k, 0]]) / 2.0
                    grad[j] += w * s * np.array([points[i, 1] - points[k, 1], points[k, 0] - points[i, 0]]) / 2.0
                    grad[k] += w * s * np.array([points[j, 1] - points[i, 1], points[i, 0] - points[j, 0]]) / 2.0
            
            points = points - esc_lr * grad
            esc_lr = max(0.0025, esc_lr * 0.96)
            
            current_min = compute_min_area(points)
            if current_min > best_min_area:
                best_min_area = current_min
                best_points = points.copy()
    
    # Restore overall best configuration
    points = best_points.copy()
    
    # Normalize to convex hull unit area
    hull = ConvexHull(points)
    scale = np.sqrt(1.0 / hull.volume)
    centroid = np.mean(points[hull.vertices], axis=0)
    points = (points - centroid) * scale + 0.5
    
    return points