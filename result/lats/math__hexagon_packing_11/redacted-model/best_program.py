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
    sqrt3 = np.sqrt(3)
    hex_height = sqrt3  # Height of a unit regular hexagon
    hex_width = 2.0      # Width of a unit regular hexagon
    
    def get_hex_vertices(x, y, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + np.deg2rad(60 * i + 30)
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            vertices.append((vx, vy))
        return np.array(vertices)
    
    def hex_distance(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return np.sqrt(dx**2 + dy**2)
    
    def compute_outer_side(vertices):
        max_proj = 0.0
        for angle in np.deg2rad([0, 60, 120]):
            proj = vertices @ np.array([np.cos(angle), np.sin(angle)])
            max_proj = max(max_proj, proj.max() - proj.min())
        return max_proj / 1.5
    
    def objective(x):
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        all_vertices = []
        for i in range(11):
            verts = get_hex_vertices(positions[i,0], positions[i,1], angles[i])
            all_vertices.extend(verts)
        all_vertices = np.array(all_vertices)
        R = compute_outer_side(all_vertices)
        return R
    
    def constraints():
        cons = []
        for i in range(11):
            for j in range(i+1, 11):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: hex_distance(x[2*i], x[2*i+1], x[2*j], x[2*j+1]) - 1.8
                })
        return cons
    
    initial_positions = np.array([
        [0, 0],
        [-1.5, 0],
        [1.5, 0],
        [-0.75, hex_height/2],
        [0.75, hex_height/2],
        [-0.75, -hex_height/2],
        [0.75, -hex_height/2],
        [-2.25, hex_height/2],
        [2.25, hex_height/2],
        [-2.25, -hex_height/2],
        [2.25, -hex_height/2],
    ])
    initial_angles = np.zeros(11)
    x0 = np.hstack([initial_positions.flatten(), initial_angles])
    
    bounds = [(-5, 5)] * 22 + [(0, 0)] * 11
    cons = constraints()
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-9, 'maxiter': 500}
    )
    
    x_opt = result.x
    positions_opt = x_opt[:22].reshape(11, 2)
    angles_opt = x_opt[22:33]
    
    all_vertices = []
    for i in range(11):
        verts = get_hex_vertices(positions_opt[i,0], positions_opt[i,1], angles_opt[i])
        all_vertices.extend(verts)
    all_vertices = np.array(all_vertices)
    outer_hex_side_length = compute_outer_side(all_vertices)
    
    inner_hex_data = np.column_stack([positions_opt, angles_opt])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length