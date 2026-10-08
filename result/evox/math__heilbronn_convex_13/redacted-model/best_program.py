import numpy as np
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    np.random.seed(42)  # Fixed seed for full determinism
    
    # Enhanced initialization: 7-5-1 ring configuration with 3-fold symmetry modulation
    # 7 boundary points (prime) naturally avoid collinear triple degeneracies
    angles_7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_radii = 1.0 * (1.0 + 0.06 * np.sin(3 * angles_7 + np.pi/12))
    outer_points = np.column_stack([outer_radii * np.cos(angles_7), outer_radii * np.sin(angles_7)])
    
    # Middle ring: 5 points staggered by π/7 rotation, radially modulated with phase shift
    angles_5 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/7
    middle_radii = 0.73 * (1.0 + 0.04 * np.sin(3 * angles_5 + np.pi/5))
    middle_points = np.column_stack([middle_radii * np.cos(angles_5), middle_radii * np.sin(angles_5)])
    
    center_point = np.array([[0.0, 0.0]])
    
    points = np.vstack([outer_points, middle_points, center_point])
    
    # Initial normalization to unit area convex hull
    try:
        hull = ConvexHull(points)
        area_scale = np.sqrt(1.0 / hull.area)
        points = points * area_scale
    except:
        pass
    
    # Efficient k-smallest triangle areas computation
    def k_smallest_areas(pts, k=12):
        """Return the k smallest triangle areas for robust optimization."""
        areas = []
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                edge_vec = pts[j] - pts[i]
                for k_idx in range(j + 1, n_pts):
                    cross = edge_vec[0] * (pts[k_idx, 1] - pts[i, 1]) - edge_vec[1] * (pts[k_idx, 0] - pts[i, 0])
                    area = 0.5 * abs(cross)
                    areas.append(area)
        areas.sort()
        return areas[:k]
    
    def min_triangle_area(pts):
        """Fast minimum triangle area computation."""
        min_area = float('inf')
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                edge_vec = pts[j] - pts[i]
                for k_idx in range(j + 1, n_pts):
                    cross = edge_vec[0] * (pts[k_idx, 1] - pts[i, 1]) - edge_vec[1] * (pts[k_idx, 0] - pts[i, 0])
                    area = 0.5 * abs(cross)
                    if area < min_area:
                        min_area = area
        return min_area
    
    # Phase 1: Aggressive hill climbing with simulated annealing acceptance
    # Uses wider exploration directions and adaptive step sizing
    best_points = points.copy()
    best_k_areas = k_smallest_areas(points, k=12)
    best_min_area = best_k_areas[0]
    step_size = 0.08
    iterations = 300
    temperature = 0.18  # Higher initial temp for escaping basins
    
    for it in range(iterations):
        improved_any = False
        point_indices = np.random.permutation(n)
        for pt_idx in point_indices:
            old_point = points[pt_idx].copy()
            improved = False
            
            # 48 directional vectors for comprehensive search coverage
            directions = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2*np.pi, 48, endpoint=False)]
            
            for dx, dy in directions:
                points[pt_idx] = old_point + step_size * np.array([dx, dy])
                
                current_k = k_smallest_areas(points, k=8)
                current_min = current_k[0]
                
                # Strict improvement acceptance
                if current_min > best_min_area + 1e-12:
                    best_min_area = current_min
                    best_k_areas = current_k
                    best_points = points.copy()
                    improved = True
                    improved_any = True
                    break
                # Simulated annealing: probabilistic escape from local minima
                elif np.random.random() < temperature * 0.015 and current_min >= best_min_area * 0.95:
                    improved = True
                    break
            
            if not improved:
                points[pt_idx] = old_point
        
        # Adaptive step size and temperature reduction
        if (it + 1) % 40 == 0:
            step_size *= 0.7
            temperature *= 0.8
        
        # Strategic restart with controlled perturbation when plateaued
        if (it + 1) % 80 == 0 and not improved_any:
            points = best_points.copy()
            points += np.random.normal(0, step_size * 0.5, points.shape)
            step_size = max(step_size, 0.015)
            temperature = min(temperature + 0.04, 0.15)
    
    # Phase 2: Lexicographic optimization with heavy weighting on smallest triangles
    # Improves near-min triangles while strictly preserving minimum area improvements
    points = best_points.copy()
    step_size = 0.035
    temperature = 0.07
    
    for it in range(180):
        for pt_idx in np.random.permutation(n):
            old_point = points[pt_idx].copy()
            
            best_score = -1
            best_candidate = old_point
            current_min = best_k_areas[0]
            
            # 44 directions for balanced precision search
            directions = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2*np.pi, 44, endpoint=False)]
            
            for dx, dy in directions:
                points[pt_idx] = old_point + step_size * np.array([dx, dy])
                
                small_areas = k_smallest_areas(points, k=12)
                
                # Hard constraint: never accept worse minimum than current best (tiny tolerance)
                if small_areas[0] >= current_min - 1e-13:
                    # Even more aggressive weighting: smallest triangle absolute dominates
                    score = small_areas[0] * 220 + sum(area * (0.82/(i+1)) for i, area in enumerate(small_areas[1:]))
                    if score > best_score:
                        best_score = score
                        best_candidate = points[pt_idx].copy()
                        if small_areas[0] > best_k_areas[0]:
                            best_k_areas = small_areas
                # Very rare probabilistic acceptance for exploration
                elif np.random.random() < 0.003 and small_areas[0] >= current_min * 0.99:
                    score = small_areas[0] * 220 + sum(area * (0.82/(i+1)) for i, area in enumerate(small_areas[1:]))
                    if score > best_score * 0.99:
                        best_score = score
                        best_candidate = points[pt_idx].copy()
            
            points[pt_idx] = best_candidate
        
        if (it + 1) % 32 == 0:
            step_size *= 0.7
            temperature *= 0.7
    
    # Phase 3: Extended fine-grained polishing passes with more samples
    step_size = 0.01
    for it in range(140):
        for pt_idx in np.random.permutation(n):
            old_point = points[pt_idx].copy()
            best_point = old_point
            current_min = min_triangle_area(points)
            
            # Increased perturbation count for finer search
            for _ in range(22):
                dx, dy = np.random.normal(0, 1, 2) * step_size
                points[pt_idx] = old_point + np.array([dx, dy])
                new_min = min_triangle_area(points)
                if new_min >= current_min - 1e-14:
                    if new_min > current_min:
                        current_min = new_min
                        best_point = points[pt_idx].copy()
                    elif np.random.random() < 0.1:
                        best_point = points[pt_idx].copy()
                        current_min = new_min
            
            points[pt_idx] = best_point
            if current_min > best_min_area:
                best_min_area = current_min
                best_points = points.copy()
    
    # Phase 4: Final unit-area normalization and range mapping
    points = best_points.copy()
    try:
        hull = ConvexHull(points)
        if hull.volume > 1e-8:  # hull.volume is area in 2D
            area_scale = np.sqrt(1.0 / hull.volume)
            centroid = np.mean(points, axis=0)
            points = centroid + (points - centroid) * area_scale
    except:
        pass
    
    # Final clean range mapping preserving proportions to [0.03, 0.97]
    coord_min = points.min(axis=0)
    coord_max = points.max(axis=0)
    coord_range = coord_max - coord_min
    if coord_range[0] > 1e-8 and coord_range[1] > 1e-8:
        points = (points - coord_min) / coord_range
        points = points * 0.94 + 0.03
    
    return points