# src/mdwater/bonds.py
import numpy as np

def compute_bond_potential(q, bond_list, k, q_eq, box_size=None):
    """
    Calculates the total bond potential energy.
    q: (N_total, 3) array of positions
    bond_list: (N_bonds, 2) array of atom index pairs
    """
    # 1. Get positions of the atoms involved in bonds
    i, j = bond_list[:, 0], bond_list[:, 1]
    
    # 2. Calculate distances
    dr = q[i] - q[j]

    # MIC (Minimum Image Convention) for Periodic boundary conditions
    if box_size is not None:
        dr = dr - box_size * np.round(dr / box_size)

    r = np.linalg.norm(dr, axis=1)
    
    # 3. Apply the formula: 1/2 * k * (r - r0)^2
    # The sum handles the sigma in your formula
    energy = 0.5 * k * np.sum((r - q_eq)**2)
    return energy

def compute_bond_forces(q, bond_list, k, q_eq, box_size=None):
    """
    Calculates the forces acting on every atom due to bonds.
    """
    n_atoms = q.shape[0]
    forces = np.zeros((n_atoms, 3))
    
    i, j = bond_list[:, 0], bond_list[:, 1]
    
    # Vectorized relative vectors and distances
    dr = q[i] - q[j]

    # MIC for Periodic boundary conditions
    if box_size is not None:
        dr = dr - box_size * np.round(dr / box_size)

    r = np.linalg.norm(dr, axis=1, keepdims=True)
    
    # Force magnitude: -k * (r - r0)
    # The (dr / r) part is the unit vector direction
    f_magnitude = -k * (r - q_eq)
    f_vectors = f_magnitude * (dr / r)
    
    # Scatter the forces back into the main array
    # np.add.at is used to handle atoms that have multiple bonds (like Oxygen)
    np.add.at(forces, i, f_vectors)
    np.add.at(forces, j, -f_vectors)
    
    return forces

