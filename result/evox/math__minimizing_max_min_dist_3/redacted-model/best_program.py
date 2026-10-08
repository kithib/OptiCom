import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    # Step 1: Generate cube vertices and face centers (14 total points)
    # Cube vertices: 8 points with coordinates (±1, ±1, ±1)
    vertices = np.array([[1, 1, 1],
                         [1, 1, -1],
                         [1, -1, 1],
                         [1, -1, -1],
                         [-1, 1, 1],
                         [-1, 1, -1],
                         [-1, -1, 1],
                         [-1, -1, -1]], dtype=np.float64)
    
    # Face centers: 6 points - one at the center of each cube face
    face_centers = np.array([[1, 0, 0],
                              [-1, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0],
                              [0, 0, 1],
                              [0, 0, -1]], dtype=np.float64)
    
    points = np.vstack([vertices, face_centers])
    
    # Normalize to unit sphere to start with bounded configuration
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Step 2: Enhanced local optimization to improve spacing
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # Compute initial distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    dmin = np.min(dist)
    dmax = np.max(dist)
    best_ratio = dmin / dmax
    best_points = points.copy()
    
    # Enhanced optimization: adaptive learning rate and more iterations
    learning_rate = 0.02
    for iteration in range(300):
        # Adaptive learning rate decay
        current_lr = learning_rate * (1 - iteration / 300)
        
        for i in range(14):
            # Compute forces: stronger repulsion from closer points
            forces = np.zeros(3)
            for j in range(14):
                if i != j:
                    vec = points[i] - points[j]
                    dist_ij = np.linalg.norm(vec)
                    if dist_ij > 0:
                        # Force inversely proportional to cube of distance
                        # Add extra repulsion for very close points
                        forces += vec / (dist_ij ** 4)
            
            # Update position with small step
            points[i] += current_lr * forces
            
            # Keep points on sphere (normalize) to maintain bounded max distance
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] /= norm
        
        # Evaluate current configuration
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        dmin = np.min(dist)
        dmax = np.max(dist)
        current_ratio = dmin / dmax
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
    
    return best_points