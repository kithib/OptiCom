import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    
    # Improved hybrid initialization: 9 boundary + 4 interior points with enhanced 3-fold symmetry
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    boundary_count = 9
    # More balanced radii distribution for interior points
    radii = np.array([0.98] * boundary_count + [0.58, 0.58, 0.42, 0.42])
    
    # Enhanced rotation pattern to avoid collinearities and improve distribution
    angles[9] += np.pi / 9.0
    angles[10] += np.pi / 9.0 + 2 * np.pi / 3.0
    angles[11] += 2 * np.pi / 9.0 + np.pi / 3.0
    angles[12] += 2 * np.pi / 9.0 + np.pi
    
    points = np.column_stack([
        radii * np.cos(angles) + 0.012 * rng.standard_normal(n),
        radii * np.sin(angles) + 0.012 * rng.standard_normal(n)
    ])
    
    # Initial normalization to unit convex hull area
    hull = ConvexHull(points)
    area = hull.volume
    scale = np.sqrt(1.0 / area)
    points = points * scale
    points = points - np.mean(points, axis=0) + 0.5
    
    # Stage 1: Enhanced L-BFGS-B optimization with adaptive LogSumExp objective
    def objective(points_flat):
        """Smoothed objective using LogSumExp with improved adaptive weighting"""
        pts = points_flat.reshape(-1, 2)
        alpha = 180.0
        beta = 250.0
        sum_exp = 0.0
        
        for i, j, k in combinations(range(n), 3):
            area = 0.5 * abs(
                (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
            )
            # Progressive penalty schedule for very small triangles
            if area < 0.003:
                sum_exp += np.exp(-beta * area) * 15.0
            elif area < 0.007:
                sum_exp += np.exp(-(alpha + 50) * area) * 5.0
            else:
                sum_exp += np.exp(-alpha * area)
        
        return (1.0 / alpha) * np.log(sum_exp / 286.0)
    
    bounds = [(0.01, 0.99)] * (2 * n)
    result = minimize(
        objective,
        points.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-9}
    )
    
    points = result.x.reshape(-1, 2)
    
    # Stage 2: Multi-phase gradient ascent with improved weighting schemes
    
    # Phase 2a: Larger steps, sharper focus on smallest triangles with inverse-cube weights
    learning_rate = 0.028
    for iteration in range(200):
        gradients = np.zeros((n, 2))
        
        triangles = []
        for combo in combinations(range(n), 3):
            i, j, k = combo
            A, B, C = points[i], points[j], points[k]
            cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
            area = abs(cross) / 2.0
            triangles.append((area, combo, cross))
        
        triangles.sort(key=lambda x: x[0])
        
        # Adaptive focus: more triangles early, fewer later
        focus_count = min(60 - iteration // 5, 45)
        for area, combo, cross in triangles[:focus_count]:
            i, j, k = combo
            # Stronger weighting: inverse cube for early iterations
            weight = 1.0 / (area + 0.0025) ** 3.0
            sign = np.sign(cross)
            
            A, B, C = points[i], points[j], points[k]
            
            gradients[i, 0] += weight * (B[1] - C[1]) * sign / 2.0
            gradients[i, 1] += weight * (C[0] - B[0]) * sign / 2.0
            gradients[j, 0] += weight * (C[1] - A[1]) * sign / 2.0
            gradients[j, 1] += weight * (A[0] - C[0]) * sign / 2.0
            gradients[k, 0] += weight * (A[1] - B[1]) * sign / 2.0
            gradients[k, 1] += weight * (B[0] - A[0]) * sign / 2.0
        
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > 1e-10:
            gradients = gradients / grad_norm
        
        points += learning_rate * gradients
        
        if iteration % 40 == 39:
            learning_rate *= 0.68
    
    # Phase 2b: Fine tuning with wider focus and smoother weights
    learning_rate = 0.012
    for iteration in range(140):
        gradients = np.zeros((n, 2))
        
        triangles = []
        for combo in combinations(range(n), 3):
            i, j, k = combo
            A, B, C = points[i], points[j], points[k]
            cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
            area = abs(cross) / 2.0
            triangles.append((area, combo, cross))
        
        triangles.sort(key=lambda x: x[0])
        
        focus_count = min(90, len(triangles))
        for area, combo, cross in triangles[:focus_count]:
            i, j, k = combo
            weight = 1.0 / (area + 0.0035) ** 1.9
            sign = np.sign(cross)
            
            A, B, C = points[i], points[j], points[k]
            
            gradients[i, 0] += weight * (B[1] - C[1]) * sign / 2.0
            gradients[i, 1] += weight * (C[0] - B[0]) * sign / 2.0
            gradients[j, 0] += weight * (C[1] - A[1]) * sign / 2.0
            gradients[j, 1] += weight * (A[0] - C[0]) * sign / 2.0
            gradients[k, 0] += weight * (A[1] - B[1]) * sign / 2.0
            gradients[k, 1] += weight * (B[0] - A[0]) * sign / 2.0
        
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > 1e-10:
            gradients = gradients / grad_norm
        
        points += learning_rate * gradients
        
        if iteration % 35 == 34:
            learning_rate *= 0.72
    
    # Phase 2c: Final refinement with dual-targeted optimization
    # First, improve the smallest triangles collectively
    for refinement_iter in range(70):
        gradients = np.zeros((n, 2))
        
        # Find the smallest N triangles and improve all of them simultaneously
        triangles = []
        for combo in combinations(range(n), 3):
            i, j, k = combo
            A, B, C = points[i], points[j], points[k]
            cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
            area = abs(cross) / 2.0
            triangles.append((area, combo, cross))
        
        triangles.sort(key=lambda x: x[0])
        
        # Focus on the smallest 15 triangles for collective improvement
        focus_size = min(15, len(triangles))
        for area, combo, cross in triangles[:focus_size]:
            i, j, k = combo
            weight = 1.0 / (area + 0.001) ** 2.5
            sign = np.sign(cross)
            
            A, B, C = points[i], points[j], points[k]
            
            gradients[i, 0] += weight * (B[1] - C[1]) * sign / 2.0
            gradients[i, 1] += weight * (C[0] - B[0]) * sign / 2.0
            gradients[j, 0] += weight * (C[1] - A[1]) * sign / 2.0
            gradients[j, 1] += weight * (A[0] - C[0]) * sign / 2.0
            gradients[k, 0] += weight * (A[1] - B[1]) * sign / 2.0
            gradients[k, 1] += weight * (B[0] - A[0]) * sign / 2.0
        
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > 1e-12:
            gradients = gradients / grad_norm
        
        points += 0.006 * gradients
    
    # Then, specifically target the single smallest triangle iteratively
    for final_iter in range(40):
        min_area = float('inf')
        worst_combo = None
        worst_cross = 0
        
        for combo in combinations(range(n), 3):
            i, j, k = combo
            A, B, C = points[i], points[j], points[k]
            cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
            area = abs(cross) / 2.0
            if area < min_area:
                min_area = area
                worst_combo = combo
                worst_cross = cross
        
        if worst_combo is not None:
            i, j, k = worst_combo
            sign = np.sign(worst_cross)
            A, B, C = points[i], points[j], points[k]
            
            grad = np.zeros((n, 2))
            grad[i, 0] = (B[1] - C[1]) * sign / 2.0
            grad[i, 1] = (C[0] - B[0]) * sign / 2.0
            grad[j, 0] = (C[1] - A[1]) * sign / 2.0
            grad[j, 1] = (A[0] - C[0]) * sign / 2.0
            grad[k, 0] = (A[1] - B[1]) * sign / 2.0
            grad[k, 1] = (B[0] - A[0]) * sign / 2.0
            
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-12:
                grad = grad / gnorm
            
            points += 0.005 * grad
    
    # Final normalization to unit convex hull area
    hull = ConvexHull(points)
    hull_area = hull.volume
    if hull_area > 0:
        scale_factor = np.sqrt(1.0 / hull_area)
        points = points * scale_factor
        points = points - points.mean(axis=0) + 0.5
    
    return points