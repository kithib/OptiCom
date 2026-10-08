import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    sqrt3 = np.sqrt(3)
    sqrt3_half = sqrt3 / 2.0
    sqrt3_quarter = sqrt3 / 4.0
    sqrt3_sixth = sqrt3 / 6.0
    sqrt3_eighth = sqrt3 / 8.0
    
    # Symmetric initialization with fine-tuned interior positions based on known Heilbronn patterns
    points = np.array([
        [0.0, 0.0],                                # Vertex 1
        [1.0, 0.0],                                # Vertex 2
        [0.5, sqrt3_half],                         # Vertex 3
        [0.335, 0.0],                              # Bottom edge point 1 (fine-tuned)
        [0.665, 0.0],                              # Bottom edge point 2 (fine-tuned)
        [0.175, sqrt3 * 0.17],                     # Left edge point (fine-tuned)
        [0.825, sqrt3 * 0.17],                     # Right edge point (fine-tuned)
        [0.325, sqrt3 * 0.34],                     # Interior point 1 (symmetric, fine-tuned)
        [0.675, sqrt3 * 0.34],                     # Interior point 2 (symmetric, fine-tuned)
        [0.5, sqrt3 * 0.155],                      # Center-bottom interior point (fine-tuned)
        [0.5, sqrt3 * 0.335],                      # Center interior point (fine-tuned)
    ], dtype=np.float64)
    
    # Ensure all points stay within the triangle boundaries
    def project_point(p):
        x, y = p
        y = np.clip(y, 0, sqrt3_half)
        x_left = y / sqrt3
        x_right = 1 - y / sqrt3
        x = np.clip(x, x_left, x_right)
        return np.array([x, y])
    
    def clamp_to_triangle(pts):
        for i, p in enumerate(pts):
            pts[i] = project_point(p)
        return pts
    
    # Compute minimal triangle area
    def triangle_area(a, b, c):
        return 0.5 * np.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    
    def min_triangle_area(pts):
        min_area = float('inf')
        min_trip = None
        for triplet in combinations(range(n), 3):
            area = triangle_area(pts[triplet[0]], pts[triplet[1]], pts[triplet[2]])
            if area < min_area:
                min_area = area
                min_trip = triplet
        return min_area, min_trip
    
    # Phase 1: Smallest triangle perturbation optimization with noise injection
    rng = np.random.default_rng(seed=42)
    phase1_iterations = 550
    
    for i in range(phase1_iterations):
        current_min, min_triplet = min_triangle_area(points)
        if min_triplet is None:
            break
        
        # Periodic noise injection with adaptive amplitude and occasional vertex relaxation
        if i > 0 and i % 20 == 0:
            noise_amplitude = 0.012 + (i / phase1_iterations) * 0.022
            # Every 4th noise cycle, relax vertex positions slightly to escape tight configurations
            start_idx = 0 if (i // 20) % 4 == 0 else 3
            for idx in range(start_idx, n):
                noise = rng.uniform(-noise_amplitude, noise_amplitude, 2)
                points[idx] = project_point(points[idx] + noise)
            continue
            
        # Higher initial learning rate with improved decay profile
        learning_rate = 0.020 * (1 - i / phase1_iterations) + 0.004
        
        # Move vertices of the smallest triangle outward slightly
        for idx in min_triplet:
            centroid = np.mean(points[list(min_triplet)], axis=0)
            direction = points[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                new_point = points[idx] + learning_rate * direction
                points[idx] = project_point(new_point)
    
    # Phase 2: Hill climbing with extended iterations and improved acceptance criterion
    current_min, _ = min_triangle_area(points)
    phase2_iterations = 22000
    initial_step = 0.07
    
    for i in range(phase2_iterations):
        step_size = initial_step * (1 - i / phase2_iterations) + 0.002
        
        num_moves = 1 if i < phase2_iterations * 0.55 else (2 if i < phase2_iterations * 0.88 else 3)
        
        idx = rng.integers(0, n)
        delta = rng.uniform(-step_size, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        if num_moves >= 2:
            idx2 = rng.integers(0, n)
            delta2 = rng.uniform(-step_size * 0.7, step_size * 0.7, 2)
            new_points[idx2] += delta2
        
        if num_moves >= 3:
            idx3 = rng.integers(0, n)
            delta3 = rng.uniform(-step_size * 0.5, step_size * 0.5, 2)
            new_points[idx3] += delta3
        
        new_points = clamp_to_triangle(new_points)
        new_min, _ = min_triangle_area(new_points)
        # Fine-tuned acceptance criterion with temperature-like probability for equal moves
        if new_min > current_min or (new_min == current_min and rng.random() < 0.25):
            points = new_points
            current_min = new_min
    
    return points