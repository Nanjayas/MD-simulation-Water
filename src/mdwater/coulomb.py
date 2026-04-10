# src/mdwater/coulomb.py
import numpy as np
from scipy.special import erfc

# ==========================================
# COULOMB POTENTIAL AND FORCES
# ==========================================
# Coulomb potential 
def compute_coulomb_potential(q, coulomb_pair_list, charges, C, box_size=None, cutoff=None):
    """Calculates the total Coulomb potential energy."""
    if len(coulomb_pair_list) == 0:
        return 0.0
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]

    # --- 1. MINIMUM IMAGE CONVENTION ---
    if box_size is not None:
        r_vec = r_vec - box_size * np.round(r_vec / box_size)
        
    r_sq = np.sum(r_vec**2, axis=1)
    
    # --- 2. CUTOFF MASK ---
    if cutoff is not None:
        mask = r_sq < (cutoff ** 2)
        r_sq = r_sq[mask]
        idx_i = idx_i[mask] # Filter charges too!
        idx_j = idx_j[mask] # Filter charges too!
        
    if len(r_sq) == 0:
        return 0.0

    r = np.sqrt(r_sq)
    
    # V = C * (q_i * q_j) / r
    energies = C * (charges[idx_i] * charges[idx_j]) / r
    
    # --- 3. CUTOFF SHIFT ---
    if cutoff is not None:
        # Calculate exactly what the Coulomb energy is at the 9.0A line
        energy_at_cutoff = C * (charges[idx_i] * charges[idx_j]) / cutoff
        # Shift the curve so it hits zero smoothly
        energies = energies - energy_at_cutoff
    
    return np.sum(energies)

# Coulomb forces:
def compute_coulomb_forces(q, coulomb_pair_list, charges, C, box_size=None, cutoff=None):
    """Calculates the Coulomb forces between pairs of atoms."""
    forces = np.zeros_like(q)
    
    if len(coulomb_pair_list) == 0:
        return forces
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]
    
    # --- 1. MINIMUM IMAGE CONVENTION ---
    if box_size is not None:
        r_vec = r_vec - box_size * np.round(r_vec / box_size)
    
    r_sq = np.sum(r_vec**2, axis=1)
    
    # --- 2. CUTOFF MASK ---
    if cutoff is not None:
        mask = r_sq < (cutoff ** 2)
        r_sq = r_sq[mask]
        r_vec = r_vec[mask]
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

    if len(r_sq) == 0:
        return forces
        
    r = np.sqrt(r_sq)
    
    # force_prefactor = C * (q_i * q_j) / r^3
    force_prefactor = (C * charges[idx_i] * charges[idx_j]) / (r_sq * r)
    
    # F_i = prefactor * (q_i - q_j)
    f_ij = force_prefactor[:, np.newaxis] * r_vec
    
    # Apply Newton's Third Law
    np.add.at(forces, idx_i, f_ij)
    np.add.at(forces, idx_j, -f_ij)
    
    return forces
#---------------------------------------
# We can't apply cutoff to Coulomb interactions like we do with Lennard-Jones because of the long-range nature of electrostatics
# We can apply Ewald summation technique
#----------------------------------------
# ==========================================
# EWALD SUMMATION FOR ELECTROSTATICS
# ==========================================
# Ewald potentials
def compute_ewald_real_potential(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff):
    """Calculates the short-range Real Space part of Ewald."""
    if len(coulomb_pair_list) == 0:
        return 0.0
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]
    
    # 1. MIC (Ghosts)
    r_vec = r_vec - box_size * np.round(r_vec / box_size)
    r_sq = np.sum(r_vec**2, axis=1)
    
    # 2. Cutoff Mask
    mask = r_sq < (cutoff ** 2)
    r_sq = r_sq[mask]
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    
    if len(r_sq) == 0:
        return 0.0
        
    r = np.sqrt(r_sq)
    
    # 3. The Ewald Real Space Formula
    # Notice we multiply by erfc(alpha * r) 
    energies = C * (charges[idx_i] * charges[idx_j]) * erfc(alpha * r) / r
    
    return np.sum(energies)

def compute_ewald_self_energy(charges, C, alpha):
    """Calculates the constant Self-Energy correction."""
    # E_self = (alpha / sqrt(pi)) * sum(q^2)
    prefactor = C * alpha / np.sqrt(np.pi)
    energy = prefactor * np.sum(charges**2)
    return energy

def compute_ewald_reciprocal_potential(q, charges, C, alpha, box_size, k_max):
    """Calculates the long-range Reciprocal Space part of Ewald."""
    V = box_size ** 3
    energy = 0.0
    
    # 1. Generate a 3D grid of integers (n_x, n_y, n_z) from -k_max to +k_max
    n_range = np.arange(-k_max, k_max + 1)
    n_x, n_y, n_z = np.meshgrid(n_range, n_range, n_range, indexing='ij')
    
    # Flatten the grids and filter out the (0, 0, 0) vector (k cannot be zero)
    n_vectors = np.column_stack((n_x.ravel(), n_y.ravel(), n_z.ravel()))
    mask = np.any(n_vectors != 0, axis=1)
    n_vectors = n_vectors[mask]
    
    # 2. Convert integers to k-vectors: k = (2 * pi / L) * n
    k_vectors = (2 * np.pi / box_size) * n_vectors
    k_sq = np.sum(k_vectors**2, axis=1)
    
    # 3. Calculate the exponential damping factor for all waves
    exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
    
    # 4. Loop over every wave and check how the charges align with it (Structure Factor)
    for i in range(len(k_vectors)):
        k = k_vectors[i]
        
        # Dot product of position and wave vector
        k_dot_r = np.dot(q, k) 
        
        # Sum of q * cos(k*r) and q * sin(k*r)
        sum_cos = np.sum(charges * np.cos(k_dot_r))
        sum_sin = np.sum(charges * np.sin(k_dot_r))
        
        # Add to total reciprocal energy
        S_k_sq = sum_cos**2 + sum_sin**2
        energy += exp_term[i] * S_k_sq
        
    # Multiply by final constants: C * (4pi / 2V)
    prefactor = C * (4 * np.pi) / (2 * V)
    return prefactor * energy

def compute_total_ewald_potential(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """The master function that replaces compute_coulomb_potential."""
    
    # alpha controls the width of the Gaussian clouds. 
    # A standard choice is ~3.0 / cutoff
    alpha = 3.0 / cutoff 
    
    # k_max controls how many waves we simulate. Usually 5 or 6 is plenty.
    k_max = 5 
    
    # Calculate the three pieces!
    e_real = compute_ewald_real_potential(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    e_recip = compute_ewald_reciprocal_potential(q, charges, C, alpha, box_size, k_max)
    e_self = compute_ewald_self_energy(charges, C, alpha)
    
    # Total Energy = Real + Reciprocal - Self
    total_coulomb_energy = e_real + e_recip - e_self
    return total_coulomb_energy
# Ewald forces
def compute_ewald_real_forces(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff):
    """Calculates the short-range Real Space forces of Ewald."""
    forces = np.zeros_like(q)
    if len(coulomb_pair_list) == 0:
        return forces
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]
    
    # 1. MIC (Ghosts)
    r_vec = r_vec - box_size * np.round(r_vec / box_size)
    r_sq = np.sum(r_vec**2, axis=1)
    
    # 2. Cutoff Mask
    mask = r_sq < (cutoff ** 2)
    r_sq = r_sq[mask]
    r_vec = r_vec[mask]
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    
    if len(r_sq) == 0:
        return forces
        
    r = np.sqrt(r_sq)
    
    # 3. The Force Prefactor (Product Rule result)
    term1 = erfc(alpha * r) / (r_sq * r)
    term2 = (2.0 * alpha / np.sqrt(np.pi)) * np.exp(-alpha**2 * r_sq) / r_sq
    
    force_prefactor = C * charges[idx_i] * charges[idx_j] * (term1 + term2)
    
    # 4. Multiply by distance vector and apply Newton's Third Law
    f_ij = force_prefactor[:, np.newaxis] * r_vec
    
    np.add.at(forces, idx_i, f_ij)
    np.add.at(forces, idx_j, -f_ij)
    
    return forces

def compute_ewald_reciprocal_forces(q, charges, C, alpha, box_size, k_max):
    """Calculates the long-range Reciprocal Space forces of Ewald."""
    V = box_size ** 3
    forces = np.zeros_like(q)
    
    # 1. Generate the same 3D wave grid
    n_range = np.arange(-k_max, k_max + 1)
    n_x, n_y, n_z = np.meshgrid(n_range, n_range, n_range, indexing='ij')
    n_vectors = np.column_stack((n_x.ravel(), n_y.ravel(), n_z.ravel()))
    mask = np.any(n_vectors != 0, axis=1)
    n_vectors = n_vectors[mask]
    
    k_vectors = (2 * np.pi / box_size) * n_vectors
    k_sq = np.sum(k_vectors**2, axis=1)
    exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
    
    prefactor = C * (4 * np.pi) / V
    
    # 2. Loop over waves to calculate gradients
    for i in range(len(k_vectors)):
        k = k_vectors[i]
        k_dot_r = np.dot(q, k)
        
        sum_cos = np.sum(charges * np.cos(k_dot_r))
        sum_sin = np.sum(charges * np.sin(k_dot_r))
        
        # Derivative of the wave structure factor
        wave_force_magnitude = charges * (np.sin(k_dot_r) * sum_cos - np.cos(k_dot_r) * sum_sin)
        
        # Multiply by the k-vector direction and add to total force
        f_wave = prefactor * exp_term[i] * np.outer(wave_force_magnitude, k)
        forces += f_wave
        
    return forces

def compute_total_ewald_forces(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """The master function that replaces compute_coulomb_forces."""
    alpha = 3.0 / cutoff 
    k_max = 5 
    
    # We only need Real and Reciprocal forces (Self force is 0!)
    f_real = compute_ewald_real_forces(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    f_recip = compute_ewald_reciprocal_forces(q, charges, C, alpha, box_size, k_max)
    
    return f_real + f_recip

# ==========================================
# PARTICLE-MESH EWALD (PME) SUMMATION
# ==========================================

def compute_pme_reciprocal_potential(q, charges, C, alpha, box_size, grid_size=32):
    """Calculates reciprocal space energy using a 3D charge grid and Fast Fourier Transforms."""
    V = box_size ** 3
    
    # 1. Map charges to a 3D grid using Numpy's histogramdd 
    # We wrap coordinates using modulo to strictly enforce the periodic box
    bins = [np.linspace(0, box_size, grid_size + 1)] * 3
    rho_grid, _ = np.histogramdd(q % box_size, bins=bins, weights=charges)
    
    # 2. Fast Fourier Transform (FFT) the 3D charge grid into Reciprocal Space
    rho_k = np.fft.fftn(rho_grid)
    
    # 3. Create the mathematical K-vectors for the grid
    kx = np.fft.fftfreq(grid_size, d=box_size/grid_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, kx, kx, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    
    # Ignore the divide-by-zero warning at the origin (k=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
        exp_term[0, 0, 0] = 0.0  # Set origin to 0 to cancel out the infinity
        
    # 4. Calculate total reciprocal energy: 1/2 * V * sum( |rho_k|^2 * exp_term )
    # (Divided by grid_size**6 because FFT is unscaled in numpy)
    energy = 0.5 * (4 * np.pi * C / V) * np.sum(np.abs(rho_k)**2 * exp_term) / (grid_size ** 6)
    
    return energy

def compute_total_pme_potential(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """The master PME function for potential energy."""
    alpha = 3.0 / cutoff 
    grid_size = 32 # A standard 32x32x32 FFT grid
    
    # Real and Self space are mathematically same than Ewals
    e_real = compute_ewald_real_potential(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    e_self = compute_ewald_self_energy(charges, C, alpha)
    
    # Reciprocal space PME is calculated using the FFT-based method
    e_recip = compute_pme_reciprocal_potential(q, charges, C, alpha, box_size, grid_size)
    
    return e_real + e_recip - e_self

def compute_pme_reciprocal_forces(q, charges, C, alpha, box_size, grid_size=32):
    """Calculates reciprocal space forces using a 3D charge grid and FFTs."""
    V = box_size ** 3
    
    # 1. Map charges to grid (Using the exact same logic as your potential function)
    bins = [np.linspace(0, box_size, grid_size + 1)] * 3
    rho_grid, edges = np.histogramdd(q % box_size, bins=bins, weights=charges)
    
    # 2. Fast Fourier Transform (FFT) of the charge grid into Reciprocal Space
    rho_k = np.fft.fftn(rho_grid)
    
    # 3. Create the mathematical K-vectors for the grid
    kx = np.fft.fftfreq(grid_size, d=box_size/grid_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, kx, kx, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
        exp_term[0, 0, 0] = 0.0
        
    # The prefactor accounts for FFT unscaled values.
    # Note: np.fft.ifftn divides by N (grid_size**3) so we adjust the scaling to match the energy.
    prefactor = (4 * np.pi * C / V) / (grid_size ** 3)
    
    # 4. Calculate Electric field components in reciprocal space: E_k = -i * k * phi_k
    E_k_x = -1j * kx * prefactor * rho_k * exp_term
    E_k_y = -1j * ky * prefactor * rho_k * exp_term
    E_k_z = -1j * kz * prefactor * rho_k * exp_term
    
    # 5. Inverse FFT to get the Electric field on the real-space grid
    E_grid_x = np.real(np.fft.ifftn(E_k_x))
    E_grid_y = np.real(np.fft.ifftn(E_k_y))
    E_grid_z = np.real(np.fft.ifftn(E_k_z))
    
    # 6. Interpolate the grid forces back to the particles
    forces = np.zeros_like(q)
    
    # Map particle positions to grid indices (reversing the histogram mapping)
    idx_x = np.clip(np.digitize((q[:, 0] % box_size), edges[0]) - 1, 0, grid_size - 1)
    idx_y = np.clip(np.digitize((q[:, 1] % box_size), edges[1]) - 1, 0, grid_size - 1)
    idx_z = np.clip(np.digitize((q[:, 2] % box_size), edges[2]) - 1, 0, grid_size - 1)
    
    # Force on particle is q * E
    forces[:, 0] = charges * E_grid_x[idx_x, idx_y, idx_z]
    forces[:, 1] = charges * E_grid_y[idx_x, idx_y, idx_z]
    forces[:, 2] = charges * E_grid_z[idx_x, idx_y, idx_z]
    
    return forces

def compute_total_pme_forces(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """The master PME function for forces."""
    alpha = 3.0 / cutoff 
    grid_size = 32 # A standard 32x32x32 FFT grid
    
    # Real space forces are mathematically the same as standard Ewald
    f_real = compute_ewald_real_forces(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    
    # Reciprocal space PME calculated using the FFT-based method
    f_recip = compute_pme_reciprocal_forces(q, charges, C, alpha, box_size, grid_size)
    
    return f_real + f_recip