# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    
    # Generate symmetric hexagonal-inspired initialization with 3-fold symmetry
    # Combines: Exemplar 1/2 structure (center + inner hex + outer hex) with current artifact's hull normalization
    points = []
    
    # Center point (1 point)
    points.append([0.5, 0.5])
    
    # Inner hexagon (6 points) - 3-fold symmetry basis
    inner_radius = 0.22
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = 0.5 + inner_radius * np.cos(angle)
        y = 0.5 + inner_radius * np.sin(angle)
        points.append([x, y])
    
    # Outer hexagon vertices (6 points) - alternating radii for variety
    outer_radii = [0.44, 0.46, 0.44, 0.46, 0.44, 0.46]
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        x = 0.5 + outer_radii[i] * np.cos(angle)
        y = 0.5 + outer_radii[i] * np.sin(angle)
        points.append([x, y])
    
    points = np.array(points)
    
    # Normalize convex hull to unit area
    hull_points = points[np.argsort(np.arctan2(points[:, 1] - 0.5, points[:, 0] - 0.5))]
    hull_area = 0.5 * np.abs(np.sum(hull_points[:-1, 0] * hull_points[1:, 1] - hull_points[1:, 0] * hull_points[:-1, 1]))
    hull_area += 0.5 * np.abs(hull_points[-1, 0] * hull_points[0, 1] - hull_points[0, 0] * hull_points[-1, 1])
    if hull_area > 0.01:
        points = (points - np.mean(points, axis=0)) / np.sqrt(hull_area) + 0.5
    
    # Hybrid optimization pipeline combining best of all sources:
    # 1. Force-based optimization (from current artifact)
    points = force_optimization(points, iterations=120, step_size=0.05)
    
    # 2. Smallest-K refinement (Exemplar 2's gradient approach)
    points = smallest_k_refinement(points, iterations=120)
    
    # 3. Smooth LogSumExp gradient refinement (from current artifact, improved)
    points = gradient_refinement(points, iterations=100)
    
    # 4. Final exact min area gradient ascent with adaptive step (Exemplar 1's final refinement)
    best_points = points.copy()
    best_min_area = 0.0
    step_size = 0.008
    
    for iteration in range(150):
        min_area, grad = compute_min_area_and_grad(points, n)
        
        if min_area > best_min_area + 1e-12:
            best_min_area = min_area
            best_points = points.copy()
            step_size *= 1.03
        else:
            step_size *= 0.97
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
        
        points += step_size * grad + rng.normal(0, 0.001, size=points.shape)
        points = np.clip(points, 0.01, 0.99)
    
    best_points = np.round(best_points, 12)
    
    return best_points.astype(np.float64)


def compute_min_area_and_grad(pts, n):
    """From Exemplar 1: Compute smallest triangle area and approximate gradient."""
    min_area = float('inf')
    grad = np.zeros_like(pts)
    
    for i, j, k in combinations(range(n), 3):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        area = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])
        
        if area < 1e-10:
            area = 1e-10
        
        if area < min_area + 1e-12:
            min_area = area
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            sign = np.sign(cross) if cross != 0 else 1.0
            grad[i] += sign * np.array([v2[1] - v1[1], v1[0] - v2[0]])
            grad[j] += sign * np.array([-v2[1], v2[0]])
            grad[k] += sign * np.array([v1[1], -v1[0]])
    
    return min_area, grad


def force_optimization(points: np.ndarray, iterations: int = 100, step_size: float = 0.05) -> np.ndarray:
    """From current artifact: Force-based repulsion optimization."""
    n = len(points)
    for _ in range(iterations):
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff ** 2) + 1e-6
                dist = np.sqrt(dist_sq)
                force_mag = step_size / (dist_sq * n)
                force = force_mag * diff / dist
                forces[i] += force
                forces[j] -= force
        points += forces
        points = np.clip(points, 0.01, 0.99)
    return points


def smallest_k_refinement(points: np.ndarray, iterations: int = 100) -> np.ndarray:
    """From Exemplar 2: Focus gradient on smallest triangles."""
    n = len(points)
    learning_rate = 0.04
    for _ in range(iterations):
        tri_indices = np.array(list(combinations(range(n), 3)))
        v1 = points[tri_indices[:, 1]] - points[tri_indices[:, 0]]
        v2 = points[tri_indices[:, 2]] - points[tri_indices[:, 0]]
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        smallest_k = max(50, len(areas) // 6)
        small_idx = np.argpartition(areas, smallest_k)[:smallest_k]
        
        gradients = np.zeros_like(points)
        for idx in small_idx:
            i, j, k = tri_indices[idx]
            area = areas[idx]
            if area < 1e-8:
                area = 1e-8
            strength = learning_rate / (area * 50 + 1)
            gradients[i] += strength * (points[i] - points[j]) + strength * (points[i] - points[k])
            gradients[j] += strength * (points[j] - points[i]) + strength * (points[j] - points[k])
            gradients[k] += strength * (points[k] - points[i]) + strength * (points[k] - points[j])
        
        points += gradients
        points = np.clip(points, 0.01, 0.99)
        learning_rate *= 0.99
    
    return points


def gradient_refinement(points: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Improved smooth LogSumExp gradient ascent."""
    n = len(points)
    alpha = 120.0
    
    for _ in range(iterations):
        grad = np.zeros_like(points)
        triples = list(combinations(range(n), 3))
        
        areas = []
        for i, j, k in triples:
            a = 0.5 * np.abs((points[j,0] - points[i,0]) * (points[k,1] - points[i,1]) - 
                               (points[j,1] - points[i,1]) * (points[k,0] - points[i,0]))
            areas.append(a)
        
        areas_arr = np.array(areas)
        weights = np.exp(-alpha * (areas_arr - np.max(areas_arr)))
        weights /= weights.sum() + 1e-10
        
        for idx, (i, j, k) in enumerate(triples):
            if weights[idx] < 1e-6:
                continue
            w = weights[idx]
            p1, p2, p3 = points[i], points[j], points[k]
            grad[i] += w * 0.5 * np.array([p2[1] - p3[1], p3[0] - p2[0]])
            grad[j] += w * 0.5 * np.array([p3[1] - p1[1], p1[0] - p3[0]])
            grad[k] += w * 0.5 * np.array([p1[1] - p2[1], p2[0] - p1[0]])
        
        points += 0.012 * grad / (np.linalg.norm(grad) + 1e-6)
        points = np.clip(points, 0.01, 0.99)
    
    return points


# EVOLVE-BLOCK-END