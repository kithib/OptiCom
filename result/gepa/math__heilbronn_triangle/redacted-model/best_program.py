import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    sqrt3 = np.sqrt(3)
    
    # Weakness 1 FIXED: Tuned interior initial positions combining best from exemplars + additional refinement
    # to reduce early near-colinear triples and provide a better starting basin
    points = np.array([
        [0.0, 0.0],                             # Vertex 1 (fixed)
        [1.0, 0.0],                             # Vertex 2 (fixed)
        [0.5, sqrt3 / 2.0],                     # Vertex 3 (fixed)
        [1/3, 0.0],                             # Edge AB: 1/3 division avoids 0.25 pattern degeneracy
        [2/3, 0.0],                             # Edge AB: 2/3
        [1/6, sqrt3 / 6.0],                     # Edge AC: 1/3 parametric
        [5/6, sqrt3 / 6.0],                     # Edge BC: 1/3 parametric
        [0.315, sqrt3 * 0.35],                  # Interior left-up: Tuned midpoint for better spacing
        [0.685, sqrt3 * 0.35],                  # Interior right-up: Symmetric counterpart
        [0.345, sqrt3 * 0.195],                 # Interior left-down: Tuned for balanced area distribution
        [0.655, sqrt3 * 0.195],                 # Interior right-down: Symmetric counterpart
    ], dtype=np.float64)
    
    # Weakness 2 FIXED: Optimized area computation with running min-area early exit
    # (stops inner loop when area2 falls below current minimum threshold)
    def compute_min_area(pts, current_min_thresh=1e-15):
        min_area2 = float('inf')
        thresh2 = current_min_thresh * 2.0  # Convert threshold to match area2 units
        for i in range(n):
            for j in range(i + 1, n):
                dx_ij = pts[j,0] - pts[i,0]
                dy_ij = pts[j,1] - pts[i,1]
                for k in range(j + 1, n):
                    area2 = abs(dx_ij * (pts[k,1] - pts[i,1]) - 
                                dy_ij * (pts[k,0] - pts[i,0]))
                    if area2 < min_area2:
                        min_area2 = area2
                        if min_area2 < thresh2:
                            return min_area2 / 2.0  # Early exit: already below threshold
        return min_area2 / 2.0
    
    # Weakness 3 FIXED: Tuned optimization loop with extended iterations, adaptive seed diversity,
    # symmetric candidate generation, phase-aware num_probes, quicker plateau handling in late phase,
    # and additional final local polish pass to squeeze out last bits of performance
    rng = np.random.default_rng(seed=314159)  # Alternative seed for broader basin exploration
    step = 0.022  # Balanced initial step between exemplars 1 and 2 for faster early progress
    current_min = compute_min_area(points)
    best_min = current_min
    best_points = points.copy()
    no_improve = 0
    
    for iteration in range(16000):  # Extended iterations for more refinement time
        # Balanced perturbation schedule: edge focus early, interior focus late with symmetry awareness
        if iteration < 5000:
            if iteration % 5 < 3:
                idx = rng.integers(3, 7)
            else:
                idx = rng.integers(7, n)
        else:
            # Late phase: focus 55% on interior points (most room for improvement)
            if iteration % 9 < 5:
                idx = rng.integers(7, n)
            elif iteration % 4 < 2:
                idx = rng.integers(3, 7)
            else:
                idx = rng.integers(3, n)  # Mixed backup
        
        # Adaptive best-of-N probing: higher probe counts at phase boundaries and late iterations
        if iteration < 6000:
            num_probes = 7 if iteration % 3 == 0 else 6
        elif iteration < 12000:
            num_probes = 8 if iteration % 4 == 0 else 7
        else:
            # Final fine-tuning phase: more probes for precision
            num_probes = 9 if iteration % 3 == 0 else 8
            
        candidates = []
        candidate_mins = []
        for _ in range(num_probes):
            cand_pt = points[idx] + rng.uniform(-step, step, 2)
            x, y = cand_pt
            y = max(y, 0.0)
            y_max1 = sqrt3 * x
            y_max2 = sqrt3 * (1 - x)
            y_max = min(y_max1, y_max2)
            if y > y_max:
                y = y_max
                x = y / sqrt3 if y_max1 < y_max2 else 1 - y / sqrt3
            cand_pt = np.array([x, y])
            tmp_pts = points.copy()
            tmp_pts[idx] = cand_pt
            candidates.append(cand_pt)
            candidate_mins.append(compute_min_area(tmp_pts, best_min))
        
        best_candidate_min = max(candidate_mins)
        best_candidate_idx_local = candidate_mins.index(best_candidate_min)
        best_candidate_pt = candidates[best_candidate_idx_local]
        
        # Accept-equal move policy with refined step adaptation and phase-aware caps
        if best_candidate_min >= current_min - 1e-14:
            if best_candidate_min > current_min + 1e-14:
                no_improve = 0
                # Tuned step increase with phase-aware cap
                if iteration < 6000:
                    step_cap = 0.085
                elif iteration < 12000:
                    step_cap = 0.065
                else:
                    step_cap = 0.035  # Lower cap for fine-tuning phase
                step = min(step * 1.065, step_cap)
            else:
                no_improve += 1
            points[idx] = best_candidate_pt
            current_min = best_candidate_min
            if current_min > best_min + 1e-14:
                best_min = current_min
                best_points = points.copy()
        else:
            no_improve += 1
        
        # Refined plateau handling: progressively earlier restarts with phase-adjusted thresholds
        if iteration < 8000:
            plateau_thresh = 70
            step_reduc = 0.92
        elif iteration < 13000:
            plateau_thresh = 55
            step_reduc = 0.90
        else:
            # Fine-tuning phase: even quicker plateau response
            plateau_thresh = 40
            step_reduc = 0.88
        if no_improve > plateau_thresh:
            min_step = 0.0007 if iteration > 13000 else 0.001
            step = max(step * step_reduc, min_step)
            no_improve = 0
            points = best_points.copy()
            current_min = best_min
    
    # Additional final polish pass: small perturbations on interior points only with high probe count
    polish_step = 0.003
    for _ in range(800):
        idx = rng.integers(7, n)  # Only polish interior points in final phase
        candidates = []
        candidate_mins = []
        for _ in range(12):
            cand_pt = best_points[idx] + rng.uniform(-polish_step, polish_step, 2)
            x, y = cand_pt
            y = max(y, 0.0)
            y_max1 = sqrt3 * x
            y_max2 = sqrt3 * (1 - x)
            y_max = min(y_max1, y_max2)
            if y > y_max:
                y = y_max
                x = y / sqrt3 if y_max1 < y_max2 else 1 - y / sqrt3
            cand_pt = np.array([x, y])
            tmp_pts = best_points.copy()
            tmp_pts[idx] = cand_pt
            candidates.append(cand_pt)
            candidate_mins.append(compute_min_area(tmp_pts, best_min))
        
        best_candidate_min = max(candidate_mins)
        if best_candidate_min > best_min + 1e-14:
            best_candidate_idx_local = candidate_mins.index(best_candidate_min)
            best_points[idx] = candidates[best_candidate_idx_local]
            best_min = best_candidate_min
    
    return best_points