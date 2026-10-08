import numpy as np
from itertools import combinations
import warnings


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute twice the area of triangle for efficiency (avoids 0.5 factor in comparisons)."""
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _min_twice_area(points: np.ndarray) -> float:
    """Compute minimum twice-area among all triangles."""
    min_twice = float('inf')
    n = len(points)
    for i, j, k in combinations(range(n), 3):
        twice = _triangle_area(points[i], points[j], points[k])
        if twice < 1e-15:  # colinear points
            return 0.0
        if twice < min_twice:
            min_twice = twice
    return min_twice


def _min_triangle_area(points: np.ndarray) -> float:
    """Compute minimum triangle area for a set of points."""
    return 0.5 * _min_twice_area(points)


def _points_in_triangle(points: np.ndarray, V0: np.ndarray, V1: np.ndarray, V2: np.ndarray) -> bool:
    """Check if all points are inside or on the boundary of the triangle."""
    for pt in points:
        v0v1 = V1 - V0
        v0v2 = V2 - V0
        A = np.column_stack([v0v1, v0v2])
        b = pt - V0
        try:
            u, v = np.linalg.solve(A, b)
            if u < -1e-10 or v < -1e-10 or u + v > 1.0 + 1e-10:
                return False
        except np.linalg.LinAlgError:
            return False
    return True


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    # Vertices of bounding equilateral triangle
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    vertices = np.array([V0, V1, V2])

    def _get_barycentric_coords(num_points: int, rng) -> np.ndarray:
        """Generate random points uniformly in barycentric coordinates."""
        points_bary = np.zeros((num_points, 3))
        for i in range(num_points):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            points_bary[i] = np.array([1 - r1 - r2, r1, r2])
        return points_bary

    def _bary2cart(bary_coords: np.ndarray) -> np.ndarray:
        """Convert barycentric to Cartesian coordinates."""
        return (bary_coords @ vertices).reshape(-1, 2)

    configs = []
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2  # Height of unit equilateral triangle

    # Config 1: Symmetric pattern (from Exemplar 2)
    config1 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, h],  # vertices
        [0.5, 0.0], [0.25, h/2], [0.75, h/2],  # edge midpoints
        [1/3, h/3], [2/3, h/3],  # lower interior
        [0.5, h/3], [0.4, 2*h/5], [0.6, 2*h/5],  # additional symmetric
    ], dtype=np.float64)[:11]
    configs.append(config1)

    # Config 2: Hexagonal pattern with vertices (from Exemplar 2)
    config2 = [V0.copy(), V1.copy(), V2.copy()]  # vertices kept fixed
    config2.append((2.0 * V0 + V1) / 3.0)
    config2.append((V0 + 2.0 * V1) / 3.0)
    config2.append((2.0 * V0 + V2) / 3.0)
    config2.append((V0 + 2.0 * V2) / 3.0)
    config2.append((2.0 * V1 + V2) / 3.0)
    config2.append((V1 + 2.0 * V2) / 3.0)
    config2.append((V0 + V1 + V2) / 3.0)
    config2.append((V0 + V1) / 2.0)
    config2 = np.array(config2, dtype=np.float64)[:11]
    configs.append(config2)

    best_points = None
    best_min_twice = 0.0
    best_min_area = 0.0

    # Eval preset configs first (from Exemplar 2)
    for cfg in configs:
        ma = _min_triangle_area(cfg)
        if ma > best_min_area:
            best_min_area = ma
            best_min_twice = ma * 2.0
            best_points = cfg.copy()

    # Add random configs (more than Exemplar 1)
    rng = np.random.default_rng(seed=42)
    for _ in range(2000):
        bary = _get_barycentric_coords(n, rng)
        pts = _bary2cart(bary)
        ma = _min_triangle_area(pts)
        if ma > best_min_area:
            best_min_area = ma
            best_min_twice = ma * 2.0
            best_points = pts.copy()

    if best_points is None:
        best_points = _bary2cart(_get_barycentric_coords(n, rng))

    # Simple hill-climbing (from Exemplar 1)
    step_size = 0.03
    for _ in range(120):
        improved = False
        for i in range(n):
            # Keep vertices fixed for stability
            if i in [0, 1, 2]:
                v0_flag = np.allclose(best_points[i], V0, atol=1e-6)
                v1_flag = np.allclose(best_points[i], V1, atol=1e-6)
                v2_flag = np.allclose(best_points[i], V2, atol=1e-6)
                if v0_flag or v1_flag or v2_flag:
                    continue
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = best_points.copy()
                    new_pts[i] += np.array([dx, dy])
                    v0v1 = V1 - V0
                    v0v2 = V2 - V0
                    A = np.column_stack([v0v1, v0v2])
                    b = new_pts[i] - V0
                    try:
                        u, v = np.linalg.solve(A, b)
                        u_clamped = np.clip(u, 0.0, 1.0)
                        v_clamped = np.clip(v, 0.0, 1.0 - u_clamped)
                        new_pts[i] = V0 + u_clamped * v0v1 + v_clamped * v0v2
                    except np.linalg.LinAlgError:
                        continue
                    current_ma = _min_triangle_area(new_pts)
                    if current_ma > best_min_area + 1e-12:
                        best_min_area = current_ma
                        best_min_twice = current_ma * 2.0
                        best_points = new_pts
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            step_size *= 0.5

    # Stochastic hill climbing (from Exemplar 2)
    rng_opt = np.random.default_rng(seed=12345)
    step = step_size * 2 if step_size > 0.001 else 0.01
    for _ in range(2000):
        for i in range(n):
            if i in [0, 1, 2]:
                v0_flag = np.allclose(best_points[i], V0, atol=1e-6)
                v1_flag = np.allclose(best_points[i], V1, atol=1e-6)
                v2_flag = np.allclose(best_points[i], V2, atol=1e-6)
                if v0_flag or v1_flag or v2_flag:
                    continue
            candidate = best_points.copy()
            candidate[i] += rng_opt.uniform(-step, step, size=2)
            v0v1 = V1 - V0
            v0v2 = V2 - V0
            A = np.column_stack([v0v1, v0v2])
            b = candidate[i] - V0
            try:
                u, v = np.linalg.solve(A, b)
                u_clamped = np.clip(u, 0.0, 1.0)
                v_clamped = np.clip(v, 0.0, 1.0 - u_clamped)
                candidate[i] = V0 + u_clamped * v0v1 + v_clamped * v0v2
            except np.linalg.LinAlgError:
                continue
            current_ma = _min_triangle_area(candidate)
            if current_ma > best_min_area + 1e-12:
                best_min_area = current_ma
                best_min_twice = current_ma * 2.0
                best_points = candidate

    # Simulated annealing (from Current Artifact)
    best_min_twice_anneal = best_min_twice
    best_points_anneal = best_points.copy()
    T0 = 0.05
    rng_restart = np.random.default_rng(seed=9999)
    for restart in range(5):
        current = best_points_anneal.copy()
        current_min = best_min_twice_anneal
        T = T0
        for iter_inner in range(800):
            for i in range(n):
                if i in [0, 1, 2]:
                    if (np.allclose(current[i], V0, atol=1e-6) or
                        np.allclose(current[i], V1, atol=1e-6) or
                        np.allclose(current[i], V2, atol=1e-6)):
                        continue
                candidate = current.copy()
                candidate[i] += rng_restart.uniform(-T*50, T*50, size=2)
                v0v1 = V1 - V0
                v0v2 = V2 - V0
                A = np.column_stack([v0v1, v0v2])
                b = candidate[i] - V0
                try:
                    u, v = np.linalg.solve(A, b)
                    u_clamped = np.clip(u, 0.0, 1.0)
                    v_clamped = np.clip(v, 0.0, 1.0 - u_clamped)
                    candidate[i] = V0 + u_clamped * v0v1 + v_clamped * v0v2
                except np.linalg.LinAlgError:
                    continue
                cand_min = _min_twice_area(candidate)
                if cand_min > current_min + 1e-12:
                    current = candidate
                    current_min = cand_min
                    if cand_min > best_min_twice_anneal + 1e-12:
                        best_min_twice_anneal = cand_min
                        best_points_anneal = candidate
            T *= 0.99

    best_points = best_points_anneal
    best_min_twice = best_min_twice_anneal

    # Final validation and correction (from Current Artifact)
    for idx in range(n):
        v0v1 = V1 - V0
        v0v2 = V2 - V0
        A = np.column_stack([v0v1, v0v2])
        b = best_points[idx] - V0
        try:
            u, v = np.linalg.solve(A, b)
            u_clamped = np.clip(u, 0.0, 1.0)
            v_clamped = np.clip(v, 0.0, 1.0 - u_clamped)
            best_points[idx] = V0 + u_clamped * v0v1 + v_clamped * v0v2
        except np.linalg.LinAlgError:
            pass

    return np.asarray(best_points, dtype=np.float64)