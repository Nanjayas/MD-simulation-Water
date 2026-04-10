# src/mdwater/lennard_jones.py
import numpy as np

# LJ potentials:
def compute_lj_potential(q, lj_pair_list, A, B, box_size=None, cutoff=None):
    """Calculates the total Lennard-Jones potential energy."""
    if len(lj_pair_list) == 0:
        return 0.0
        
    idx_i = lj_pair_list[:, 0]
    idx_j = lj_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]

    # MIC (Minimum Image Convention)for Periodic boundary conditions
    if box_size is not None:
        r_vec = r_vec - box_size * np.round(r_vec / box_size)

    r_sq = np.sum(r_vec**2, axis=1)

    # CUTOFF
    if cutoff is not None:
        mask = r_sq < (cutoff ** 2)
        r_sq = r_sq[mask] # Filter out the atoms that are too far away
        
    # If no atoms are within the cutoff, return 0
    if len(r_sq) == 0:
        return 0.0
    
    # Calculate 1/r^6 and 1/r^12 efficiently (on the filtered atoms)
    r_6 = 1.0 / (r_sq ** 3)
    r_12 = r_6 ** 2
    
    # V = A/r^12 - B/r^6
    energies = (A * r_12) - (B * r_6)

    # --- THE CUTOFF SHIFT ---
    if cutoff is not None:
        # Calculate the exact energy AT the cutoff distance
        cutoff_6 = 1.0 / (cutoff ** 6)
        cutoff_12 = cutoff_6 ** 2
        energy_at_cutoff = (A * cutoff_12) - (B * cutoff_6)
        
        # Shift all energies so they smoothly hit zero!
        energies = energies - energy_at_cutoff
    
    return np.sum(energies)

# LJ forces:
def compute_lj_forces(q, lj_pair_list, A, B, box_size=None, cutoff=None):
    """
    Calculates the Lennard-Jones forces between pairs of atoms.
    """
    forces = np.zeros_like(q)
    
    if len(lj_pair_list) == 0:
        return forces
        
    # Get the indices of the pairs
    idx_i = lj_pair_list[:, 0]
    idx_j = lj_pair_list[:, 1]
    
    # Calculate the distance vectors (q_i - q_j)
    r_vec = q[idx_i] - q[idx_j]

    # MIC
    if box_size is not None:
        r_vec = r_vec - box_size * np.round(r_vec / box_size)
    
    # Calculate r^2 (avoiding square root for performance)
    # keepdims=True ensures the shape is (N, 1) so it broadcasts correctly with r_vec
    #r_sq = np.sum(r_vec**2, axis=1, keepdims=True)

    # keepdims=False for now so the mask is a simple 1D array
    r_sq = np.sum(r_vec**2, axis=1)

    # The cotoff mask:
    if cutoff is not None:
        mask = r_sq < (cutoff ** 2)
        r_sq = r_sq[mask]
        r_vec = r_vec[mask]
        idx_i = idx_i[mask]   # Filter the indices too
        idx_j = idx_j[mask]   # Filter the indices too

    if len(r_sq) == 0:
        return forces

    # Reshape r_sq back to (N, 1) for broadcasting with r_vec
    r_sq = r_sq.reshape(-1, 1)
    
    # Calculate r^-8 and r^-14
    r_8 = 1.0 / (r_sq ** 4)
    r_14 = 1.0 / (r_sq ** 7)
    
    # F_magnitude = (12A / r^14) - (6B / r^8)
    force_magnitude = (12.0 * A * r_14) - (6.0 * B * r_8)
    
    # F_i = F_magnitude * (q_i - q_j)
    f_ij = force_magnitude * r_vec
    
    # Apply Newton's Third Law
    np.add.at(forces, idx_i, f_ij)
    np.add.at(forces, idx_j, -f_ij)
    
    return forces