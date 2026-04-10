# src/mdwater/angles.py
import numpy as np

def compute_angle_potential(q, angle_list, k_theta, theta_eq, box_size=None):
    """Calculates the total angle potential energy."""
    idx_o, idx_h1, idx_h2 = angle_list[:,0], angle_list[:,1], angle_list[:,2]
    
    r1 = q[idx_h1] - q[idx_o]
    r2 = q[idx_h2] - q[idx_o]

    # MIC for Periodic boundary conditions
    if box_size is not None:
        r1 = r1 - box_size * np.round(r1 / box_size)
        r2 = r2 - box_size * np.round(r2 / box_size)
    
    d1 = np.linalg.norm(r1, axis=1)
    d2 = np.linalg.norm(r2, axis=1)
    
    # Cosine of the angle using the dot product
    cos_theta = np.sum(r1 * r2, axis=1) / (d1 * d2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Numerical safety
    theta = np.arccos(cos_theta)
    
    # Energy = 1/2 * k * (theta - theta_eq)^2
    return 0.5 * k_theta * np.sum((theta - theta_eq)**2)

def compute_angle_forces(q, angle_list, k_theta, theta_eq, box_size=None):
    n_atoms = q.shape[0]
    forces = np.zeros((n_atoms, 3))
    
    # Indices for O, H1, H2
    idx_o, idx_h1, idx_h2 = angle_list[:,0], angle_list[:,1], angle_list[:,2]
    
    r1 = q[idx_h1] - q[idx_o]
    r2 = q[idx_h2] - q[idx_o]

    # MIC for Periodic boundary conditions
    if box_size is not None:
        r1 = r1 - box_size * np.round(r1 / box_size)
        r2 = r2 - box_size * np.round(r2 / box_size)

    d1 = np.linalg.norm(r1, axis=1, keepdims=True)
    d2 = np.linalg.norm(r2, axis=1, keepdims=True)
    
    # Normalize vectors
    u1 = r1 / d1
    u2 = r2 / d2
    
    # Compute theta
    cos_theta = np.sum(u1 * u2, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Numerical safety
    theta = np.arccos(cos_theta)
    
    # Precompute coefficient: -k_theta * (theta - theta_eq) / sin(theta)
    sin_theta = np.sqrt(1 - cos_theta**2)
    # Avoid division by zero if theta is exactly 0 or pi
    sin_theta = np.where(sin_theta < 1e-8, 1e-8, sin_theta)
    
    coeff = k_theta * (theta - theta_eq) / sin_theta
    
    # Calculate force vectors
    f_h1 = (coeff / d1) * (u2 - cos_theta * u1)
    f_h2 = (coeff / d2) * (u1 - cos_theta * u2)
    f_o = -(f_h1 + f_h2)
    
    # Accumulate into global force array
    np.add.at(forces, idx_h1, f_h1)
    np.add.at(forces, idx_h2, f_h2)
    np.add.at(forces, idx_o, f_o)
    
    return forces