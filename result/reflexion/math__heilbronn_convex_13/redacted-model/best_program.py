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
    rng = np.random.default_rng(seed=42)
    
    # Enhanced configuration: 7 boundary + 6 interior with highly refined 3-fold symmetry
    # Carefully balanced initialization to maximize minimum triangle area
    points = []
    
    # 7 boundary points forming a convex heptagon - highly optimized perturbations
    n_boundary = 7
    boundary_scale = 1.0
    # Extensively tuned angle offsets for optimal boundary spacing
    angle_offsets = np.array([0.0, 0.020, -0.016, 0.024, -0.020, 0.012, -0.020])
    for i in range(n_boundary):
        theta = 2 * np.pi * i / n_boundary + angle_offsets[i]
        points.append([boundary_scale * np.cos(theta), boundary_scale * np.sin(theta)])
    
    # First interior ring (3 points) - highly optimized radius and stagger
    radius1 = 0.538
    offset1 = np.pi / 4.30  # ~41.86 degrees stagger, carefully adjusted
    for i in range(3):
        theta = 2 * np.pi * i / 3 + offset1 + 0.014 * i
        points.append([radius1 * np.cos(theta), radius1 * np.sin(theta)])
    
    # Second interior ring (3 points) - highly optimized radius and stagger
    radius2 = 0.788
    offset2 = np.pi / 8.50  # ~21.18 degrees stagger, carefully adjusted
    for i in range(3):
        theta = 2 * np.pi * i / 3 + offset2 - 0.012 * i
        points.append([radius2 * np.cos(theta), radius2 * np.sin(theta)])
    
    points = np.array(points[:n])
    
    # Efficiently compute min triangle area for 13 points (286 triangles total)
    def compute_min_area(pts):
        """Compute area of smallest triangle formed by any 3 points"""
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = 0.5 * abs(
                        (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                    )
                    if area < min_area:
                        min_area = area
        return min_area
    
    # Compute convex hull area (for normalization and constraints)
    def compute_hull_area(pts):
        """Compute area of convex hull of points"""
        try:
            hull = ConvexHull(pts)
            return hull.volume  # In 2D, volume is the area
        except:
            # Degenerate case - compute bounding box area
            return (np.max(pts[:, 0]) - np.min(pts[:, 0])) * \
                   (np.max(pts[:, 1]) - np.min(pts[:, 1])) + 1e-10
    
    # Enhanced objective: ultra-sharp LogSumExp with multi-level weighting for maximum improvement
    def compute_smoothed_objective(pts, alpha=-320.0):
        """Smooth approximation to min area using LogSumExp with highly enhanced weighting"""
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = 0.5 * abs(
                        (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                    )
                    areas.append(area + 1e-12)  # Avoid log(0)
        
        areas = np.array(areas)
        # Primary LogSumExp for minimum with extreme alpha for ultra-sharp approximation
        log_sum = np.log(np.sum(np.exp(alpha * areas)))
        smooth_min = log_sum / alpha
        
        # Add secondary weighting on smallest 18 areas for balanced improvement
        areas_sorted = np.sort(areas)
        k_smallest = min(18, len(areas_sorted))
        secondary = np.mean(areas_sorted[:k_smallest]) * 0.52
        
        # Add tertiary weighting on smallest 7 areas for focused smallest triangle boost
        tertiary = np.mean(areas_sorted[:7]) * 0.18
        
        # Add quaternary weighting on the single smallest area for maximum emphasis
        quaternary = areas_sorted[0] * 0.08
        
        return smooth_min + secondary + tertiary + quaternary
    
    # Optimization: improve the configuration
    current_points = points.copy()
    
    # Enhanced gradient ascent - more iterations and highly adaptive learning rate
    num_iterations = 1000
    learning_rate = 0.026
    epsilon_grad = 1e-8
    
    for iteration in range(num_iterations):
        current_obj = compute_smoothed_objective(current_points)
        gradients = np.zeros_like(current_points)
        
        for i in range(n):
            for dim in range(2):
                test_pts = current_points.copy()
                test_pts[i, dim] += epsilon_grad
                new_obj = compute_smoothed_objective(test_pts)
                gradients[i, dim] = (new_obj - current_obj) / epsilon_grad
        
        current_points = current_points + learning_rate * gradients
        
        # Adaptive learning rate - slightly slower decay for better convergence
        if (iteration + 1) % 50 == 0:
            learning_rate *= 0.76
    
    # Enhanced fine-tuning with ultra-broad search - highly extended iterations
    current_min = compute_min_area(current_points)
    step_size = 0.028
    for _ in range(600):
        improved = False
        # Even more extended search range for maximum exploration
        for i in range(n):
            for dx in [-step_size*4.8, -step_size*4.2, -step_size*3.6, -step_size*3.0, -step_size*2.4, -step_size*1.8, -step_size*1.2, -step_size, -step_size/2, -step_size/3, -step_size/4, 0, 
                       step_size/4, step_size/3, step_size/2, step_size, step_size*1.2, step_size*1.8, step_size*2.4, step_size*3.0, step_size*3.6, step_size*4.2, step_size*4.8]:
                for dy in [-step_size*4.8, -step_size*4.2, -step_size*3.6, -step_size*3.0, -step_size*2.4, -step_size*1.8, -step_size*1.2, -step_size, -step_size/2, -step_size/3, -step_size/4, 0, 
                           step_size/4, step_size/3, step_size/2, step_size, step_size*1.2, step_size*1.8, step_size*2.4, step_size*3.0, step_size*3.6, step_size*4.2, step_size*4.8]:
                    if dx == 0 and dy == 0:
                        continue
                    test_points = current_points.copy()
                    test_points[i] = test_points[i] + [dx, dy]
                    new_min = compute_min_area(test_points)
                    if new_min > current_min:
                        current_points = test_points
                        current_min = new_min
                        improved = True
        if not improved:
            step_size *= 0.34
        if step_size < 0.00028:
            break
    
    # Additional optimization round: stochastic perturbation with increased range and iterations
    rng_fine = np.random.default_rng(seed=12345)
    for _ in range(500):
        i = rng_fine.integers(0, n)
        perturb = rng_fine.uniform(-0.080, 0.080, size=2)
        test_points = current_points.copy()
        test_points[i] = test_points[i] + perturb
        new_min = compute_min_area(test_points)
        if new_min > current_min:
            current_points = test_points
            current_min = new_min
    
    # Second pass fine-tuning with smaller steps and extended iterations
    current_min = compute_min_area(current_points)
    step_size = 0.0075
    for _ in range(350):
        improved = False
        for i in range(n):
            for dx in [-step_size*5.5, -step_size*4.8, -step_size*4.0, -step_size*3.2, -step_size*2.4, -step_size*1.6, -step_size, -step_size/2, 0, 
                       step_size/2, step_size, step_size*1.6, step_size*2.4, step_size*3.2, step_size*4.0, step_size*4.8, step_size*5.5]:
                for dy in [-step_size*5.5, -step_size*4.8, -step_size*4.0, -step_size*3.2, -step_size*2.4, -step_size*1.6, -step_size, -step_size/2, 0, 
                           step_size/2, step_size, step_size*1.6, step_size*2.4, step_size*3.2, step_size*4.0, step_size*4.8, step_size*5.5]:
                    if dx == 0 and dy == 0:
                        continue
                    test_points = current_points.copy()
                    test_points[i] = test_points[i] + [dx, dy]
                    new_min = compute_min_area(test_points)
                    if new_min > current_min:
                        current_points = test_points
                        current_min = new_min
                        improved = True
        if not improved:
            step_size *= 0.41
        if step_size < 0.00011:
            break
    
    # Third pass: finer optimization with extended iterations and range
    current_min = compute_min_area(current_points)
    step_size = 0.0022
    for _ in range(280):
        improved = False
        for i in range(n):
            for dx in [-step_size*9, -step_size*8, -step_size*7, -step_size*6, -step_size*5, -step_size*4, -step_size*3, -step_size*2, -step_size, 0, 
                       step_size, step_size*2, step_size*3, step_size*4, step_size*5, step_size*6, step_size*7, step_size*8, step_size*9]:
                for dy in [-step_size*9, -step_size*8, -step_size*7, -step_size*6, -step_size*5, -step_size*4, -step_size*3, -step_size*2, -step_size, 0, 
                           step_size, step_size*2, step_size*3, step_size*4, step_size*5, step_size*6, step_size*7, step_size*8, step_size*9]:
                    if dx == 0 and dy == 0:
                        continue
                    test_points = current_points.copy()
                    test_points[i] = test_points[i] + [dx, dy]
                    new_min = compute_min_area(test_points)
                    if new_min > current_min:
                        current_points = test_points
                        current_min = new_min
                        improved = True
        if not improved:
            step_size *= 0.37
        if step_size < 0.000045:
            break
    
    # Fourth pass: ultra-fine micro-optimization with further extended range and iterations
    current_min = compute_min_area(current_points)
    step_size = 0.00055
    for _ in range(220):
        improved = False
        for i in range(n):
            for dx in [-step_size*11, -step_size*9, -step_size*7, -step_size*5, -step_size*4, -step_size*3, -step_size*2, -step_size, 0, 
                       step_size, step_size*2, step_size*3, step_size*4, step_size*5, step_size*7, step_size*9, step_size*11]:
                for dy in [-step_size*11, -step_size*9, -step_size*7, -step_size*5, -step_size*4, -step_size*3, -step_size*2, -step_size, 0, 
                           step_size, step_size*2, step_size*3, step_size*4, step_size*5, step_size*7, step_size*9, step_size*11]:
                    if dx == 0 and dy == 0:
                        continue
                    test_points = current_points.copy()
                    test_points[i] = test_points[i] + [dx, dy]
                    new_min = compute_min_area(test_points)
                    if new_min > current_min:
                        current_points = test_points
                        current_min = new_min
                        improved = True
        if not improved:
            step_size *= 0.44
        if step_size < 0.000011:
            break
    
    # Fifth pass: nano-optimization for absolute final refinement
    current_min = compute_min_area(current_points)
    step_size = 0.000018
    for _ in range(150):
        improved = False
        for i in range(n):
            for dx in [-step_size*15, -step_size*12, -step_size*10, -step_size*8, -step_size*5, -step_size*3, -step_size*2, -step_size, 0, 
                       step_size, step_size*2, step_size*3, step_size*5, step_size*8, step_size*10, step_size*12, step_size*15]:
                for dy in [-step_size*15, -step_size*12, -step_size*10, -step_size*8, -step_size*5, -step_size*3, -step_size*2, -step_size, 0, 
                           step_size, step_size*2, step_size*3, step_size*5, step_size*8, step_size*10, step_size*12, step_size*15]:
                    if dx == 0 and dy == 0:
                        continue
                    test_points = current_points.copy()
                    test_points[i] = test_points[i] + [dx, dy]
                    new_min = compute_min_area(test_points)
                    if new_min > current_min:
                        current_points = test_points
                        current_min = new_min
                        improved = True
        if not improved:
            step_size *= 0.50
        if step_size < 0.000003:
            break
    
    # Final normalization: convex hull area = 1.0
    hull_area = compute_hull_area(current_points)
    area_scale = 1.0 / hull_area
    centroid = np.mean(current_points, axis=0)
    current_points = (current_points - centroid) * np.sqrt(area_scale) + centroid
    
    # Translate to ensure all coordinates are positive (for numerical stability)
    min_coords = np.min(current_points, axis=0)
    current_points = current_points - min_coords + 0.025
    
    return current_points