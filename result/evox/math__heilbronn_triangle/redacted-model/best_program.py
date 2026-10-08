import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # Calculate triangle area from three 2D points
    def triangle_area(a, b, c):
        return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    
    # Calculate minimum triangle area - optimized with early exit
    def min_area(points, current_best=None):
        min_val = float('inf')
        for combo in combinations(points, 3):
            area = triangle_area(*combo)
            if area < min_val:
                min_val = area
                if min_val < 1e-12:  # Early exit for degenerate triangles
                    return min_val
                if current_best is not None and min_val <= current_best:
                    return min_val
        return min_val if min_val != float('inf') else 0.0
    
    # Project point back into the equilateral triangle if outside using proper clamping
    def project_to_triangle(pt):
        x, y = pt
        h = np.sqrt(3) / 2
        # Clamp to valid bounds
        y = np.clip(y, 1e-10, h - 1e-10)
        x = np.clip(x, 1e-10, 1 - 1e-10)
        # Ensure below left edge y = sqrt(3)*x
        if y > np.sqrt(3) * x + 1e-10:
            y = np.sqrt(3) * x - 1e-10
        # Ensure below right edge y = sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x) + 1e-10:
            y = np.sqrt(3) * (1 - x) - 1e-10
        y = max(y, 1e-10)
        return np.array([x, y])
    
    n = 11
    h = np.sqrt(3) / 2  # Height of equilateral triangle with side length 1
    
    # Enhanced pool of high-quality initial configurations including known near-optimal layouts
    def get_initial_configs():
        configs = []
        
        # Config 1: Symmetric edge subdivision (base configuration)
        config1 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [1/3, 0.0], [2/3, 0.0], [1/6, h/3],
            [5/6, h/3], [1/3, 2*h/3], [2/3, 2*h/3],
            [0.5, h/2], [0.5, h/4],
        ])
        configs.append(config1)
        
        # Config 2: Interior center emphasis with balanced vertical spacing
        config2 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.25, 0.0], [0.75, 0.0], [0.2, h/4],
            [0.8, h/4], [0.4, h/2], [0.6, h/2],
            [0.5, h/3], [0.5, 2*h/3],
        ])
        configs.append(config2)
        
        # Config 3: Hexagonal-inspired interior tiling with even dispersion
        config3 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.1667, 0.0], [0.8333, 0.0], [0.25, 0.2887],
            [0.75, 0.2887], [0.3333, 0.5774], [0.6667, 0.5774],
            [0.5, 0.1925], [0.5, 0.6736],
        ])
        configs.append(config3)
        
        # Config 4: Dispersed collinearity-avoidant lattice with perturbed symmetry
        config4 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.2, 0.0], [0.8, 0.0], [0.18, 0.3],
            [0.82, 0.3], [0.38, 0.62], [0.62, 0.62],
            [0.5, 0.22], [0.5, 0.7],
        ])
        configs.append(config4)
        
        # Config 5: Edge-balanced 5-point base arrangement with staggered interior
        config5 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.2, 0.0], [0.8, 0.0],  # Bottom edge: 5 points including vertices
            [0.12, h/4], [0.88, h/4],  # Lower interior
            [0.28, h/2], [0.72, h/2],  # Mid interior
            [0.4, 3*h/4], [0.6, 3*h/4],  # Upper interior
        ])
        configs.append(config5)
        
        # Config 6: Near-hexagonal tiling variant with alternating offsets
        config6 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.125, 0.0], [0.875, 0.0],
            [0.2, h/5], [0.8, h/5],
            [0.33, 2*h/5], [0.67, 2*h/5],
            [0.42, 3*h/5], [0.58, 3*h/5],
        ])
        configs.append(config6)
        
        # Config 7: Asymmetric dispersion targeting known Heilbronn configurations
        config7 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.143, 0.0], [0.857, 0.0],
            [0.214, 0.185], [0.786, 0.185],
            [0.357, 0.370], [0.643, 0.370],
            [0.429, 0.555], [0.571, 0.555],
        ])
        configs.append(config7)
        
        # Config 8: Optimized boundary + interior spread configuration
        config8 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.11, 0.0], [0.89, 0.0],
            [0.14, h/3], [0.86, h/3],
            [0.28, 2*h/3], [0.72, 2*h/3],
            [0.5, h/6], [0.5, 5*h/6],
        ])
        configs.append(config8)
        
        # Config 9: Alternating row offset configuration
        config9 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.3, 0.0], [0.7, 0.0],
            [0.17, 0.22], [0.83, 0.22],
            [0.37, 0.44], [0.63, 0.44],
            [0.27, 0.66], [0.73, 0.66],
        ])
        configs.append(config9)
        
        # Config 10: Enhanced boundary optimization with asymmetric interior
        config10 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.18, 0.0], [0.82, 0.0],
            [0.13, 0.20], [0.87, 0.20],
            [0.31, 0.40], [0.69, 0.40],
            [0.39, 0.60], [0.61, 0.60],
        ])
        configs.append(config10)
        
        # Config 11: Vertically staggered asymmetric arrangement
        config11 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.22, 0.0], [0.78, 0.0],
            [0.16, 0.25], [0.84, 0.25],
            [0.34, 0.50], [0.66, 0.50],
            [0.46, 0.75], [0.54, 0.75],
        ])
        configs.append(config11)
        
        # Config 12: Boundary-optimized asymmetric arrangement with tighter spacing
        config12 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.15, 0.0], [0.85, 0.0],
            [0.11, 0.16], [0.89, 0.16],
            [0.26, 0.32], [0.74, 0.32],
            [0.41, 0.48], [0.59, 0.48],
        ])
        configs.append(config12)
        
        # Config 13: High-asymmetry layout targeting local optima avoidance
        config13 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.19, 0.0], [0.81, 0.0],
            [0.14, 0.18], [0.86, 0.22],
            [0.29, 0.38], [0.71, 0.42],
            [0.44, 0.58], [0.56, 0.62],
        ])
        configs.append(config13)
        
        # Config 14: Boundary-emphasis layout with more points on edges
        config14 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.25, 0.0], [0.75, 0.0],  # Bottom edge
            [0.3, 0.6], [0.7, 0.6],  # Upper left/right edges
            [0.22, 0.32], [0.78, 0.32],  # Mid edges
            [0.4, 0.2], [0.6, 0.45],  # Interior
        ])
        configs.append(config14)
        
        # Config 15: Enhanced asymmetric layout targeting boundary + interior balance
        config15 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.175, 0.0], [0.825, 0.0],
            [0.125, 0.19], [0.875, 0.23],
            [0.275, 0.39], [0.725, 0.43],
            [0.425, 0.59], [0.575, 0.63],
        ])
        configs.append(config15)
        
        # Config 16: Nearly symmetric hexagonal-inspired with slight asymmetry
        config16 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.2, 0.0], [0.8, 0.0],
            [0.15, 0.21], [0.85, 0.21],
            [0.3, 0.42], [0.7, 0.42],
            [0.45, 0.63], [0.55, 0.63],
        ])
        configs.append(config16)
        
        # Config 17: Optimized from previous best runs with refined boundary spacing
        config17 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.165, 0.0], [0.835, 0.0],
            [0.135, 0.205], [0.865, 0.205],
            [0.285, 0.41], [0.715, 0.41],
            [0.435, 0.615], [0.565, 0.615],
        ])
        configs.append(config17)
        
        # Config 18: Hybrid configuration with mixed hexagonal and grid properties
        config18 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.185, 0.0], [0.815, 0.0],
            [0.115, 0.175], [0.885, 0.175],
            [0.315, 0.35], [0.685, 0.35],
            [0.465, 0.525], [0.535, 0.525],
        ])
        configs.append(config18)
        
        # Config 19: Ultra-refined near-optimal from advanced heuristics
        config19 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.158, 0.0], [0.842, 0.0],
            [0.128, 0.198], [0.872, 0.198],
            [0.278, 0.396], [0.722, 0.396],
            [0.428, 0.594], [0.572, 0.594],
        ])
        configs.append(config19)
        
        # Config 20: Near-lattice boundary-optimized with staggered interior
        config20 = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.172, 0.0], [0.828, 0.0],
            [0.142, 0.228], [0.858, 0.228],
            [0.302, 0.456], [0.698, 0.456],
            [0.462, 0.684], [0.538, 0.684],
        ])
        configs.append(config20)
        
        return configs
    
    initial_configs = get_initial_configs()
    best_global_points = initial_configs[0].copy()
    best_global_min = min_area(best_global_points)
    
    # Enhanced direction set with hexagonal symmetry and varied step magnitudes
    def get_directions(step):
        # 6 hexagonal + 4 cardinal + 4 diagonal + 4 fine directions + 8 intermediate angular directions
        return [
            (-step, 0), (step, 0), (0, -step), (0, step),  # Cardinal
            (-step*0.866, -step*0.5), (-step*0.866, step*0.5),  # 60° hexagonal
            (step*0.866, -step*0.5), (step*0.866, step*0.5),    # 60° hexagonal
            (-step*0.5, -step*0.866), (-step*0.5, step*0.866),  # 120° hexagonal
            (step*0.5, -step*0.866), (step*0.5, step*0.866),    # 120° hexagonal
            (-step*0.707, -step*0.707), (-step*0.707, step*0.707),  # Diagonal 45°
            (step*0.707, -step*0.707), (step*0.707, step*0.707),    # Diagonal 45°
            (-step*0.4, 0), (step*0.4, 0), (0, -step*0.4), (0, step*0.4),  # Fine steps
            (-step*0.924, -step*0.383), (-step*0.924, step*0.383),  # 22.5° directions
            (step*0.924, -step*0.383), (step*0.924, step*0.383),    # for finer angular resolution
            (-step*0.383, -step*0.924), (-step*0.383, step*0.924),  # 67.5° directions
            (step*0.383, -step*0.924), (step*0.383, step*0.924),    # complementary angles
        ]
    
    # Enhanced coordinated move: shift two points in opposite directions to improve symmetry
    def coordinated_move(points, idx1, idx2, step):
        candidates = []
        dirs1 = get_directions(step)
        dirs2 = get_directions(step)
        # Try coordinated opposite moves for balance
        for (dx1, dy1), (dx2, dy2) in zip(dirs1[:12], dirs2[:12]):
            candidate = points.copy()
            candidate[idx1] += np.array([dx1, dy1])
            candidate[idx1] = project_to_triangle(candidate[idx1])
            candidate[idx2] += np.array([-dx2, -dy2])
            candidate[idx2] = project_to_triangle(candidate[idx2])
            candidates.append(candidate)
        return candidates
    
    # Three-point coordinated move for better exploration
    def three_point_coordinated_move(points, idx1, idx2, idx3, step):
        candidates = []
        dirs = get_directions(step * 0.7)
        # Small coordinated displacements among triplets
        for dx, dy in dirs[:8]:
            candidate = points.copy()
            # Cyclic displacement pattern
            candidate[idx1] += np.array([dx, dy])
            candidate[idx2] += np.array([-dy * 0.866 + dx * 0.5, dx * 0.866 + dy * 0.5])
            candidate[idx3] += np.array([dy * 0.866 - dx * 0.5, -dx * 0.866 - dy * 0.5])
            for i in [idx1, idx2, idx3]:
                candidate[i] = project_to_triangle(candidate[i])
            candidates.append(candidate)
        return candidates
    
    # Four-point cyclic move targeting symmetric groups
    def four_point_cyclic_move(points, idx1, idx2, idx3, idx4, step):
        candidates = []
        dirs = get_directions(step * 0.5)
        for dx, dy in dirs[:6]:
            candidate = points.copy()
            candidate[idx1] += np.array([dx, dy])
            candidate[idx2] += np.array([-dy, dx])
            candidate[idx3] += np.array([-dx, -dy])
            candidate[idx4] += np.array([dy, -dx])
            for i in [idx1, idx2, idx3, idx4]:
                candidate[i] = project_to_triangle(candidate[i])
            candidates.append(candidate)
        return candidates
    
    # Five-point coordinated move for deep refinement of interior arrangements
    def five_point_star_move(points, center_idx, ring_indices, step):
        candidates = []
        dirs = get_directions(step * 0.45)
        for dx, dy in dirs[:6]:
            candidate = points.copy()
            # Central point displacement
            candidate[center_idx] += np.array([dx * 0.35, dy * 0.35])
            # Ring points with rotational displacement
            angles = [0, np.pi*2/5, np.pi*4/5, np.pi*6/5, np.pi*8/5]
            for idx, angle in zip(ring_indices, angles[:len(ring_indices)]):
                rx = dx * np.cos(angle) - dy * np.sin(angle)
                ry = dx * np.sin(angle) + dy * np.cos(angle)
                candidate[idx] += np.array([rx * 0.65, ry * 0.65])
            candidate[center_idx] = project_to_triangle(candidate[center_idx])
            for idx in ring_indices:
                candidate[idx] = project_to_triangle(candidate[idx])
            candidates.append(candidate)
        return candidates
    
    # Explore multiple initial configurations with enhanced restart strategy
    for config_idx, initial in enumerate(initial_configs):
        # Increased restart count with varied perturbation scales for broader basin exploration
        for restart in range(22):
            if restart > 0 or config_idx > 0:
                current_points = initial.copy()
                # Varied perturbation scales based on config and restart - enhanced range
                perturb_scale = 0.028 + (restart * 0.013) + (config_idx * 0.009)
                # Allow small vertex perturbations for first 10 restarts of each config
                vertex_perturb = restart < 10
                for i in range(n):
                    if i < 3 and not vertex_perturb:
                        continue
                    current_points[i] += np.random.normal(0, perturb_scale, 2)
                    current_points[i] = project_to_triangle(current_points[i])
            else:
                current_points = initial.copy()
            
            current_min = min_area(current_points)
            
            # Extended step schedule with more granular ultra-fine tail steps
            steps = [0.125, 0.105, 0.088, 0.072, 0.058, 0.046, 0.036, 0.028, 0.021, 0.0155, 0.0115, 0.0085, 0.0062, 0.0044, 0.0031, 0.0022, 0.00155, 0.0011, 0.00078, 0.00055, 0.00038, 0.00026, 0.00018, 0.00012]
            
            for step_idx, step in enumerate(steps):
                improved = True
                iterations = 0
                # Enhanced adaptive iteration budget - more iterations for finer steps
                if step > 0.05:
                    max_iter = 85
                elif step > 0.02:
                    max_iter = 130
                elif step > 0.01:
                    max_iter = 185
                elif step > 0.005:
                    max_iter = 265
                elif step > 0.002:
                    max_iter = 370
                elif step > 0.0008:
                    max_iter = 495
                else:
                    max_iter = 680  # Ultra-fine steps get maximum iterations for deep convergence
                
                while improved and iterations < max_iter:
                    improved = False
                    # Randomize point order each cycle for better exploration
                    point_indices = np.random.permutation(n)
                    for i in point_indices:
                        # Refined vertex freezing logic - later freezing for better boundary optimization
                        if i < 3:
                            if step < 0.0022 and step_idx > 13:
                                continue
                            # Adaptive effective step for vertices for balanced stability/exploration
                            eff_step = step * 0.31 if step_idx > 9 else (step * 0.54 if step_idx > 6 else (step * 0.75 if step_idx > 3 else step * 0.92))
                            dirs = get_directions(eff_step)
                        else:
                            dirs = get_directions(step)
                        
                        for dx, dy in dirs:
                            candidate = current_points.copy()
                            candidate[i] += np.array([dx, dy])
                            candidate[i] = project_to_triangle(candidate[i])
                            
                            new_min = min_area(candidate, current_min)
                            # Accept only strict improvements with small epsilon buffer
                            if new_min > current_min + 1e-10:
                                current_min = new_min
                                current_points = candidate
                                improved = True
                                break  # Accept first good direction per point for faster convergence
                    
                    # After single-point moves, try coordinated moves for paired points at finer steps
                    if not improved and step < 0.017 and iterations < max_iter // 2:
                        # Try coordinated moves on symmetric pairs (excluding vertices)
                        pairs = [(3, 4), (5, 6), (7, 8), (9, 10), (5, 7), (6, 8), (3, 9), (4, 10)]
                        for idx1, idx2 in pairs:
                            if idx1 >= n or idx2 >= n:
                                continue
                            coord_candidates = coordinated_move(current_points, idx1, idx2, step * 0.64)
                            for candidate in coord_candidates:
                                new_min = min_area(candidate, current_min)
                                if new_min > current_min + 1e-10:
                                    current_min = new_min
                                    current_points = candidate
                                    improved = True
                                    break
                            if improved:
                                break
                    
                    # Try three-point coordinated moves at intermediate steps
                    if not improved and step < 0.01 and step > 0.00055 and iterations < max_iter // 3:
                        triplets = [(3, 5, 7), (4, 6, 8), (5, 7, 9), (6, 8, 10), (3, 7, 9), (4, 8, 10), (5, 6, 9), (7, 8, 10)]
                        for idx1, idx2, idx3 in triplets:
                            if idx3 >= n:
                                continue
                            triplet_candidates = three_point_coordinated_move(current_points, idx1, idx2, idx3, step * 0.50)
                            for candidate in triplet_candidates:
                                new_min = min_area(candidate, current_min)
                                if new_min > current_min + 1e-10:
                                    current_min = new_min
                                    current_points = candidate
                                    improved = True
                                    break
                            if improved:
                                break
                    
                    # Try four-point cyclic moves at finer steps
                    if not improved and step < 0.0045 and step > 0.00035 and iterations < max_iter // 4:
                        quads = [(3, 5, 8, 10), (4, 6, 7, 9), (3, 6, 8, 9), (4, 5, 7, 10), (5, 6, 7, 8)]
                        for idx1, idx2, idx3, idx4 in quads:
                            if idx4 >= n:
                                continue
                            quad_candidates = four_point_cyclic_move(current_points, idx1, idx2, idx3, idx4, step * 0.44)
                            for candidate in quad_candidates:
                                new_min = min_area(candidate, current_min)
                                if new_min > current_min + 1e-10:
                                    current_min = new_min
                                    current_points = candidate
                                    improved = True
                                    break
                            if improved:
                                break
                    
                    # Try five-point star moves at ultra-fine steps for deep interior refinement
                    if not improved and step < 0.002 and step > 0.0002 and iterations < max_iter // 5:
                        center_candidates = [9, 8, 7, 6, 5]
                        ring_configs = [
                            (9, [3, 4, 5, 6]),
                            (8, [5, 6, 7, 10]),
                            (7, [3, 5, 8, 9]),
                            (6, [4, 6, 7, 10]),
                            (5, [3, 4, 7, 8]),
                        ]
                        for center_idx, ring_indices in ring_configs:
                            if center_idx >= n or max(ring_indices) >= n:
                                continue
                            star_candidates = five_point_star_move(current_points, center_idx, ring_indices, step * 0.38)
                            for candidate in star_candidates:
                                new_min = min_area(candidate, current_min)
                                if new_min > current_min + 1e-10:
                                    current_min = new_min
                                    current_points = candidate
                                    improved = True
                                    break
                            if improved:
                                break
                    iterations += 1
            
            # Update global best if current config-restart yields improvement
            if current_min > best_global_min + 1e-10:
                best_global_min = current_min
                best_global_points = current_points
    
    return best_global_points.astype(np.float64)