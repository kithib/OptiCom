# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Hexagon geometry constants
    sqrt3 = np.sqrt(3)
    hex_height = sqrt3  # Distance between opposite sides of unit hexagon
    hex_width = 2.0  # Distance between opposite vertices of unit hexagon
    
    def get_hex_vertices(x, y, angle_deg):
        """Get vertices of a unit regular hexagon"""
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            vertex_angle = angle_rad + np.deg2rad(60 * i + 30)
            vx = x + np.cos(vertex_angle)
            vy = y + np.sin(vertex_angle)
            vertices.append((vx, vy))
        return np.array(vertices)
    
    def compute_min_distance_between_hexagons(x1, y1, x2, y2):
        """Compute approximate minimum distance between two unit hexagons (both 0 rotation)"""
        dx = x2 - x1
        dy = y2 - y1
        # Hexagon distance metric for axis-aligned hexagons
        adx = np.abs(dx)
        ady = np.abs(dy)
        # Transform to hex coordinates
        q = (2.0/3.0 * adx)
        r = (1.0/3.0 * adx + (sqrt3/3.0) * ady
        return max(q, r, q + r) - 1.0
    
    def objective(params):
        """Objective: minimize outer hex side length R"""
        R = params[0]
        penalty = 0.0
        
        # Check containment for each inner hexagon
        for i in range(11):
            x = params[1 + 3*i]
            y = params[2 + 3*i]
            # Check outer hexagon constraints (axis-aligned, centered at origin)
            # For outer hexagon, constraints in axial coordinates
            # Convert to hex coordinates
            q = (2.0/3.0 * x
            r = (-1.0/3.0 * x + (sqrt3/3.0) * y
            s = -q - r
            # Check hex distance from origin for all three axes
            max_qrs = max(np.abs(q), np.abs(r), np.abs(s))
            # Add penalty if hexagon extends beyond outer hexagon
            ext = max_qrs - (R - 1.0)
            if ext > 0:
                penalty += 1000.0 * ext * ext
        
        # Check non-overlap constraints
        for i in range(11):
            x1 = params[1 + 3*i]
            y1 = params[2 + 3*i]
            for j in range(i + 1, 11):
                x2 = params[1 + 3*j]
                y2 = params[2 + 3*j]
                dist = compute_min_distance_between_hexagons(x1, y1, x2, y2)
                if dist < 0:
                    penalty += 1000.0 * (-dist) * (-dist)
        
        return R + penalty
    
    # Initialize solution near known good packing for 11 hexagons
    sqrt3 = np.sqrt(3)
    initial_guess = np.array([
        4.0,  # Initial outer hex side length
        # 11 inner hexagons positions
        0.0, 0.0, 0.0,
        -1.5, 0.0, 0.0,
        1.5, 0.0, 0.0,
        -0.75, 1.5*sqrt3/2, 0.0,
        0.75, 1.5*sqrt3/2, 0.0,
        -0.75, -1.5*sqrt3/2, 0.0,
        0.75, -1.5*sqrt3/2, 0.0,
        -2.25, 0.75*sqrt3, 0.0,
        2.25, 0.75*sqrt3, 0.0,
        -2.25, -0.75*sqrt3, 0.0,
        2.25, -0.75*sqrt3, 0.0,
    ])
    
    # Optimize
    bounds = [(0.1, 10.0)] + [(-10.0, 10.0), (-10.0, 10.0), (0.0, 0.0)] * 11
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8))
    
    opt_params = result.x
    outer_hex_side_length = opt_params[0]
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = opt_params[1 + 3*i]
        inner_hex_data[i, 1] = opt_params[2 + 3*i]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
)
    bounds = [(0.1, 10.0)] + [(-10.0, 10.0), (-10.0, 10.0), (0.0, 0.0)] * 11
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8))
    
    opt_params = result.x
    outer_hex_side_length = opt_params[0]
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = opt_params[1 + 3*i]
        inner_hex_data[i, 1] = opt_params[2 + 3*i]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
10.0, 10.0), (-10.0, 10.0), (0.0, 0.0)] * 11
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8))
    
    opt_params = result.x
    outer_hex_side_length = opt_params[0]
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = opt_params[1 + 3*i]
        inner_hex_data[i, 1] = opt_params[2 + 3*i]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
)
    bounds = [(0.1, 10.0)] + [(-10.0, 10.0), (-10.0, 10.0), (0.0, 0.0)] * 11
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8))
    
    opt_params = result.x
    outer_hex_side_length = opt_params[0]
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = opt_params[1 + 3*i]
        inner_hex_data[i, 1] = opt_params[2 + 3*i]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END