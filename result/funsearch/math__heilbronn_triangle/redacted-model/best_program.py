import numpy as np
from itertools import combinations


def _in_equilateral_triangle(p: np.ndarray) -> bool:
    """Check if a point is inside or on the boundary of the equilateral triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    if y < 0 or y > h:
        return False
    if x < y / np.sqrt(3) or x > 1 - y / np.sqrt(3):
        return False
    return True


def _project_to_triangle(p: np.ndarray) -> np.ndarray:
    """Project a point back into the triangle bounds."""
    x, y = p
    h = np.sqrt(3) / 2
    y = np.clip(y, 0, h)
    x = np.clip(x, y / np.sqrt(3), 1 - y / np.sqrt(3))
    return np.array([x, y])


def _compute_min_area(points: np.ndarray) -> float:
    """Compute the minimum area of any triangle formed by the points."""
    min_area = float('inf')
    for triplet in combinations(points, 3):
        a, b, c = triplet
        area = 0.5 * np.abs(
            a[0] * (b[1] - c[1]) +
            b[0] * (c[1] - a[1]) +
            c[0] * (a[1] - b[1])
        )
        if area < min_area:
            min_area = area
    return min_area


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    np.random.seed(42)
    n = 11
    bounds = np.array([[0.0, 1.0], [0.0, np.sqrt(3) / 2]])
    
    # Phase 1: Expanded multi-restart hybrid initialization
    # Combine: structured grid candidates (Exemplar1) + maximin candidates (Exemplar2)
    best_points = None
    best_min_area = -1
    
    # Phase 1a: Try Exemplar 1's structured grid base candidates (5 attempts with varying perturbation)
    for attempt in range(5):
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [0.25, 0.0],
            [0.75, 0.0],
            [0.125, np.sqrt(3)/8],
            [0.875, np.sqrt(3)/8],
            [0.375, np.sqrt(3)/8],
            [0.625, np.sqrt(3)/8],
            [0.25, np.sqrt(3)/4],
            [0.75, np.sqrt(3)/4]
        ], dtype=np.float64)
        
        perturb_range = 0.015 + attempt * 0.005
        for i in range(3, n):
            for _ in range(15):
                candidate = points[i] + np.random.uniform(-perturb_range, perturb_range, 2)
                if _in_equilateral_triangle(candidate):
                    points[i] = candidate
                    break
        
        min_area = _compute_min_area(points)
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = points.copy()
    
    # Phase 1b: Try Exemplar 2's maximin base candidates (5 additional attempts)
    for attempt in range(5):
        points = np.zeros((n, 2))
        points[0] = [0.0, 0.0]
        points[1] = [1.0, 0.0]
        points[2] = [0.5, np.sqrt(3) / 2]
        
        for i in range(3, n):
            best_candidate = None
            best_candidate_dist = -1
            for _ in range(200):
                candidate = np.random.uniform(bounds[:, 0], bounds[:, 1], size=2)
                if _in_equilateral_triangle(candidate):
                    min_dist = np.min(np.linalg.norm(points[:i] - candidate, axis=1))
                    if min_dist > best_candidate_dist:
                        best_candidate_dist = min_dist
                        best_candidate = candidate
            points[i] = best_candidate
        
        current_min = _compute_min_area(points)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
    
    # Phase 2: 8-directional hill climbing with PROJECTION (from Current Artifact)
    # Combined: Exemplar 1's iteration budget + Current Artifact's projection (not skip)
    step_size = 0.05
    for _ in range(300):
        improved = False
        for i in range(n):
            for dx, dy in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size),
                           (-step_size*0.7, -step_size*0.7), (step_size*0.7, -step_size*0.7),
                           (-step_size*0.7, step_size*0.7), (step_size*0.7, step_size*0.7)]:
                new_points = best_points.copy()
                candidate = new_points[i] + np.array([dx, dy])
                
                # Use projection instead of skipping for better exploration (Current Artifact trait)
                if not _in_equilateral_triangle(candidate):
                    candidate = _project_to_triangle(candidate)
                
                new_points[i] = candidate
                current_min_area = _compute_min_area(new_points)
                
                if current_min_area > best_min_area:
                    best_min_area = current_min_area
                    best_points = new_points
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step_size *= 0.85
            if step_size < 0.001:
                break
    
    # Phase 3: Enhanced Random walk fine-tuning (2000 steps from Current Artifact)
    step = 0.01
    for _ in range(2000):
        idx = np.random.randint(0, n)
        orig_pt = best_points[idx].copy()
        delta = np.random.uniform(-step, step, 2)
        new_pt = orig_pt + delta
        
        new_pt = _project_to_triangle(new_pt)
        best_points[idx] = new_pt
        new_min = _compute_min_area(best_points)
        if new_min >= best_min_area:
            best_min_area = new_min
        else:
            best_points[idx] = orig_pt
    
    # Phase 4: 9-direction fine-grained polish (Combined: Current's finer initial step + Exemplar1's lower threshold)
    step_size = 0.005
    for iteration in range(200):
        improved = False
        for i in range(n):
            original = best_points[i].copy()
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    candidate = original + np.array([dx, dy])
                    if _in_equilateral_triangle(candidate):
                        best_points[i] = candidate
                        new_min = _compute_min_area(best_points)
                        if new_min > best_min_area:
                            best_min_area = new_min
                            improved = True
                            original = candidate.copy()
                        else:
                            best_points[i] = original.copy()
        if not improved:
            step_size *= 0.5
            if step_size < 0.0003:
                break
    
    return best_points.astype(np.float64)