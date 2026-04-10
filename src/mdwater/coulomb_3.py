# -*- coding: utf-8 -*-
# src/mdwater/coulomb_3.py
import numpy as np
from scipy.special import erfc

# ========== GPU ACCELERATION SWITCH ==========
USE_GPU = False   # Set to True to use GPU for Ewald reciprocal forces , False for CPU
# ============================================

# ==========================================
# COULOMB POTENTIAL AND FORCES (original)
# ==========================================
def compute_coulomb_potential(q, coulomb_pair_list, charges, C, box_size=None, cutoff=None):
    """Calculates the total Coulomb potential energy."""
    if len(coulomb_pair_list) == 0:
        return 0.0
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]

    if box_size is not None:
        r_vec = r_vec - box_size * np.round(r_vec / box_size)
        
    r_sq = np.sum(r_vec**2, axis=1)
    
    if cutoff is not None:
        mask = r_sq < (cutoff ** 2)
        r_sq = r_sq[mask]
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        
    if len(r_sq) == 0:
        return 0.0

    r = np.sqrt(r_sq)
    energies = C * (charges[idx_i] * charges[idx_j]) / r
    
    if cutoff is not None:
        energy_at_cutoff = C * (charges[idx_i] * charges[idx_j]) / cutoff
        energies = energies - energy_at_cutoff
    
    return np.sum(energies)

def compute_coulomb_forces(q, coulomb_pair_list, charges, C, box_size=None, cutoff=None):
    """Calculates the Coulomb forces between pairs of atoms."""
    forces = np.zeros_like(q)
    
    if len(coulomb_pair_list) == 0:
        return forces
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]
    
    if box_size is not None:
        r_vec = r_vec - box_size * np.round(r_vec / box_size)
    
    r_sq = np.sum(r_vec**2, axis=1)
    
    if cutoff is not None:
        mask = r_sq < (cutoff ** 2)
        r_sq = r_sq[mask]
        r_vec = r_vec[mask]
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

    if len(r_sq) == 0:
        return forces
        
    r = np.sqrt(r_sq)
    force_prefactor = (C * charges[idx_i] * charges[idx_j]) / (r_sq * r)
    f_ij = force_prefactor[:, np.newaxis] * r_vec
    
    np.add.at(forces, idx_i, f_ij)
    np.add.at(forces, idx_j, -f_ij)
    
    return forces

# ==========================================
# EWALD SUMMATION (original)
# ==========================================
def compute_ewald_real_potential(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff):
    """Calculates the short-range Real Space part of Ewald."""
    if len(coulomb_pair_list) == 0:
        return 0.0
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]
    r_vec = r_vec - box_size * np.round(r_vec / box_size)
    r_sq = np.sum(r_vec**2, axis=1)
    mask = r_sq < (cutoff ** 2)
    r_sq = r_sq[mask]
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    
    if len(r_sq) == 0:
        return 0.0
        
    r = np.sqrt(r_sq)
    energies = C * (charges[idx_i] * charges[idx_j]) * erfc(alpha * r) / r
    return np.sum(energies)

def compute_ewald_self_energy(charges, C, alpha):
    """Calculates the constant Self-Energy correction."""
    prefactor = C * alpha / np.sqrt(np.pi)
    energy = prefactor * np.sum(charges**2)
    return energy

def compute_ewald_reciprocal_potential(q, charges, C, alpha, box_size, k_max):
    """Calculates the long-range Reciprocal Space part of Ewald (CPU version)."""
    V = box_size ** 3
    energy = 0.0
    
    n_range = np.arange(-k_max, k_max + 1)
    n_x, n_y, n_z = np.meshgrid(n_range, n_range, n_range, indexing='ij')
    n_vectors = np.column_stack((n_x.ravel(), n_y.ravel(), n_z.ravel()))
    mask = np.any(n_vectors != 0, axis=1)
    n_vectors = n_vectors[mask]
    
    k_vectors = (2 * np.pi / box_size) * n_vectors
    k_sq = np.sum(k_vectors**2, axis=1)
    exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
    
    for i in range(len(k_vectors)):
        k = k_vectors[i]
        k_dot_r = np.dot(q, k)
        sum_cos = np.sum(charges * np.cos(k_dot_r))
        sum_sin = np.sum(charges * np.sin(k_dot_r))
        S_k_sq = sum_cos**2 + sum_sin**2
        energy += exp_term[i] * S_k_sq
        
    prefactor = C * (4 * np.pi) / (2 * V)
    return prefactor * energy

def compute_total_ewald_potential(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """Master function that replaces compute_coulomb_potential for Ewald."""
    alpha = 3.0 / cutoff 
    k_max = 5 
    e_real = compute_ewald_real_potential(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    e_recip = compute_ewald_reciprocal_potential(q, charges, C, alpha, box_size, k_max)
    e_self = compute_ewald_self_energy(charges, C, alpha)
    return e_real + e_recip - e_self

def compute_ewald_real_forces(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff):
    """Calculates the short-range Real Space forces of Ewald."""
    forces = np.zeros_like(q)
    if len(coulomb_pair_list) == 0:
        return forces
        
    idx_i = coulomb_pair_list[:, 0]
    idx_j = coulomb_pair_list[:, 1]
    
    r_vec = q[idx_i] - q[idx_j]
    r_vec = r_vec - box_size * np.round(r_vec / box_size)
    r_sq = np.sum(r_vec**2, axis=1)
    mask = r_sq < (cutoff ** 2)
    r_sq = r_sq[mask]
    r_vec = r_vec[mask]
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    
    if len(r_sq) == 0:
        return forces
        
    r = np.sqrt(r_sq)
    term1 = erfc(alpha * r) / (r_sq * r)
    term2 = (2.0 * alpha / np.sqrt(np.pi)) * np.exp(-alpha**2 * r_sq) / r_sq
    force_prefactor = C * charges[idx_i] * charges[idx_j] * (term1 + term2)
    f_ij = force_prefactor[:, np.newaxis] * r_vec
    np.add.at(forces, idx_i, f_ij)
    np.add.at(forces, idx_j, -f_ij)
    return forces

# ==========================================
# EWALD RECIPROCAL FORCES (with GPU switch)
# ==========================================
# ==========================================
# EWALD RECIPROCAL FORCES (with GPU switch)
# ==========================================
# CPU version (original) ¨C unchanged
def _compute_ewald_reciprocal_forces_cpu(q, charges, C, alpha, box_size, k_max):
    """CPU implementation of reciprocal space forces."""
    V = box_size ** 3
    forces = np.zeros_like(q)
    
    n_range = np.arange(-k_max, k_max + 1)
    n_x, n_y, n_z = np.meshgrid(n_range, n_range, n_range, indexing='ij')
    n_vectors = np.column_stack((n_x.ravel(), n_y.ravel(), n_z.ravel()))
    mask = np.any(n_vectors != 0, axis=1)
    n_vectors = n_vectors[mask]
    
    k_vectors = (2 * np.pi / box_size) * n_vectors
    k_sq = np.sum(k_vectors**2, axis=1)
    exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
    prefactor = C * (4 * np.pi) / V
    
    for i in range(len(k_vectors)):
        k = k_vectors[i]
        k_dot_r = np.dot(q, k)
        sum_cos = np.sum(charges * np.cos(k_dot_r))
        sum_sin = np.sum(charges * np.sin(k_dot_r))
        wave_force_magnitude = charges * (np.sin(k_dot_r) * sum_cos - np.cos(k_dot_r) * sum_sin)
        f_wave = prefactor * exp_term[i] * np.outer(wave_force_magnitude, k)
        forces += f_wave
    return forces

# GPU implementation using CuPy (replaces Numba CUDA)
if USE_GPU:
    try:
        import cupy as cp
        _GPU_EWALD_AVAILABLE = True
    except ImportError:
        _GPU_EWALD_AVAILABLE = False
        print("Warning: CuPy not installed. Ewald GPU acceleration disabled. Falling back to CPU.")

    if _GPU_EWALD_AVAILABLE:
        def _compute_ewald_reciprocal_forces_gpu(q, charges, C, alpha, box_size, k_max):
            """GPU-accelerated reciprocal space forces using CuPy."""
            #print("?? GPU kernel called for Ewald reciprocal forces (CuPy version)")
            # Precompute k-vectors and exp_term on CPU
            n_range = np.arange(-k_max, k_max + 1)
            n_x, n_y, n_z = np.meshgrid(n_range, n_range, n_range, indexing='ij')
            n_vectors = np.column_stack((n_x.ravel(), n_y.ravel(), n_z.ravel()))
            mask = np.any(n_vectors != 0, axis=1)
            n_vectors = n_vectors[mask]
            k_vectors = (2 * np.pi / box_size) * n_vectors
            k_sq = np.sum(k_vectors**2, axis=1)
            exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
            prefactor = C * (4 * np.pi) / (box_size**3)

            # Transfer to GPU
            q_gpu = cp.asarray(q, dtype=cp.float64)
            charges_gpu = cp.asarray(charges, dtype=cp.float64)
            k_vectors_gpu = cp.asarray(k_vectors, dtype=cp.float64)
            exp_term_gpu = cp.asarray(exp_term, dtype=cp.float64)
            prefactor_gpu = cp.float64(prefactor)

            # Compute structure factor S(k) = sum_j q_j * exp(i k¡¤r_j)
            # phase = k¡¤r_j, shape (M, N)
            phase = cp.dot(k_vectors_gpu, q_gpu.T)   # (M, N)
            # S_k = sum_j q_j * exp(i phase)
            S_k = cp.sum(charges_gpu[None, :] * cp.exp(1j * phase), axis=1)   # (M,)

            # Compute forces: F_i = 2 * C * (4¦Ð/V) * q_i * sum_k [ k * exp_term * Im( S_k * exp(-i k¡¤r_i) ) ]
            # First compute Im( S_k * exp(-i k¡¤r_i) ) for all i,k
            # term = exp(-i k¡¤r_i)   shape (M, N)
            term = cp.exp(-1j * phase)   # (M, N)
            # S_k * term -> (M, N)
            product = S_k[:, None] * term   # (M, N)
            Im_part = cp.imag(product)     # (M, N)

            # Force contribution per k: 2 * prefactor_gpu * exp_term_gpu * k_vector * Im_part
            # Expand dimensions: k_vectors_gpu (M,3) -> (M,1,3) ; Im_part (M,N) -> (M,N,1)
            force_contrib = (2.0 * prefactor_gpu) * exp_term_gpu[:, None, None] * k_vectors_gpu[:, None, :] * Im_part[:, :, None]
            # Sum over k
            forces_gpu = cp.sum(force_contrib, axis=0)   # (N,3)
            # Multiply by charge
            forces_gpu *= charges_gpu[:, None]

            return forces_gpu.get()

        # Public function
        def compute_ewald_reciprocal_forces(q, charges, C, alpha, box_size, k_max):
            return _compute_ewald_reciprocal_forces_gpu(q, charges, C, alpha, box_size, k_max)
    else:
        # Fallback to CPU
        def compute_ewald_reciprocal_forces(q, charges, C, alpha, box_size, k_max):
            return _compute_ewald_reciprocal_forces_cpu(q, charges, C, alpha, box_size, k_max)
else:
    # CPU-only fallback
    def compute_ewald_reciprocal_forces(q, charges, C, alpha, box_size, k_max):
        return _compute_ewald_reciprocal_forces_cpu(q, charges, C, alpha, box_size, k_max)

def compute_total_ewald_forces(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """Master function that replaces compute_coulomb_forces for Ewald."""
    alpha = 3.0 / cutoff 
    k_max = 5 
    f_real = compute_ewald_real_forces(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    f_recip = compute_ewald_reciprocal_forces(q, charges, C, alpha, box_size, k_max)
    return f_real + f_recip

# ==========================================
# PARTICLE-MESH EWALD (PME) SUMMATION
# ==========================================
def compute_pme_reciprocal_potential(q, charges, C, alpha, box_size, grid_size=32):
    """Calculates reciprocal space energy using a 3D charge grid and Fast Fourier Transforms."""
    V = box_size ** 3
    bins = [np.linspace(0, box_size, grid_size + 1)] * 3
    rho_grid, _ = np.histogramdd(q % box_size, bins=bins, weights=charges)
    rho_k = np.fft.fftn(rho_grid)
    kx = np.fft.fftfreq(grid_size, d=box_size/grid_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, kx, kx, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
    exp_term[0, 0, 0] = 0.0
    energy = 0.5 * (4 * np.pi * C / V) * np.sum(np.abs(rho_k)**2 * exp_term) / (grid_size ** 6)
    return energy

def compute_total_pme_potential(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """Master PME function for potential energy."""
    alpha = 3.0 / cutoff 
    grid_size = 32
    e_real = compute_ewald_real_potential(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    e_self = compute_ewald_self_energy(charges, C, alpha)
    e_recip = compute_pme_reciprocal_potential(q, charges, C, alpha, box_size, grid_size)
    return e_real + e_recip - e_self

# --- PME reciprocal forces (CPU version, kept for fallback) ---
def _compute_pme_reciprocal_forces_cpu(q, charges, C, alpha, box_size, grid_size=32):
    """CPU implementation of PME reciprocal forces."""
    V = box_size ** 3
    bins = [np.linspace(0, box_size, grid_size + 1)] * 3
    rho_grid, edges = np.histogramdd(q % box_size, bins=bins, weights=charges)
    rho_k = np.fft.fftn(rho_grid)
    kx = np.fft.fftfreq(grid_size, d=box_size/grid_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, kx, kx, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    exp_term = np.exp(-k_sq / (4 * alpha**2)) / k_sq
    exp_term[0, 0, 0] = 0.0
    prefactor = (4 * np.pi * C / V) / (grid_size ** 3)
    E_k_x = -1j * kx * prefactor * rho_k * exp_term
    E_k_y = -1j * ky * prefactor * rho_k * exp_term
    E_k_z = -1j * kz * prefactor * rho_k * exp_term
    E_grid_x = np.real(np.fft.ifftn(E_k_x))
    E_grid_y = np.real(np.fft.ifftn(E_k_y))
    E_grid_z = np.real(np.fft.ifftn(E_k_z))
    forces = np.zeros_like(q)
    idx_x = np.clip(np.digitize((q[:, 0] % box_size), edges[0]) - 1, 0, grid_size - 1)
    idx_y = np.clip(np.digitize((q[:, 1] % box_size), edges[1]) - 1, 0, grid_size - 1)
    idx_z = np.clip(np.digitize((q[:, 2] % box_size), edges[2]) - 1, 0, grid_size - 1)
    forces[:, 0] = charges * E_grid_x[idx_x, idx_y, idx_z]
    forces[:, 1] = charges * E_grid_y[idx_x, idx_y, idx_z]
    forces[:, 2] = charges * E_grid_z[idx_x, idx_y, idx_z]
    return forces

# --- PME reciprocal forces (GPU version with caching) ---
if USE_GPU:
    try:
        import cupy as cp
        from cupyx.scipy.fft import fftn, ifftn
        _GPU_PME_AVAILABLE = True
    except ImportError:
        _GPU_PME_AVAILABLE = False
        print("Warning: CuPy not installed. PME GPU acceleration disabled. Falling back to CPU.")

    if _GPU_PME_AVAILABLE:
        # Cache for static data
        _pme_gpu_cache = {
            'box_size': None,
            'grid_size': None,
            'alpha': None,
            'C': None,
            'kx': None, 'ky': None, 'kz': None,
            'exp_term': None,
            'prefactor': None,
        }

        def _compute_pme_reciprocal_forces_gpu(q, charges, C, alpha, box_size, grid_size=32):
            """GPU-accelerated PME reciprocal forces using CuPy with caching."""
            #print("?? GPU kernel called for PME reciprocal forces")
            global _pme_gpu_cache
            cache = _pme_gpu_cache

            # Check if cached data is still valid
            if (cache['box_size'] != box_size or cache['grid_size'] != grid_size or
                cache['alpha'] != alpha or cache['C'] != C):
                # Recompute static data on GPU
                box = cp.float64(box_size)
                grid = grid_size
                kx = cp.fft.fftfreq(grid, d=box/grid) * 2 * cp.pi
                kx, ky, kz = cp.meshgrid(kx, kx, kx, indexing='ij')
                k_sq = kx**2 + ky**2 + kz**2
                exp_term = cp.exp(-k_sq / (4 * alpha**2)) / k_sq
                exp_term[0,0,0] = 0.0
                V = box**3
                prefactor = (4 * cp.pi * C / V) / (grid ** 3)

                # Store in cache
                cache['box_size'] = box_size
                cache['grid_size'] = grid_size
                cache['alpha'] = alpha
                cache['C'] = C
                cache['kx'] = kx
                cache['ky'] = ky
                cache['kz'] = kz
                cache['exp_term'] = exp_term
                cache['prefactor'] = prefactor
            else:
                kx = cache['kx']
                ky = cache['ky']
                kz = cache['kz']
                exp_term = cache['exp_term']
                prefactor = cache['prefactor']

            # Convert inputs to GPU
            q_gpu = cp.asarray(q, dtype=cp.float64)
            charges_gpu = cp.asarray(charges, dtype=cp.float64)
            box = cp.float64(box_size)
            grid = grid_size

            # 1. Map charges to grid using integer indices
            idx = cp.floor((q_gpu % box) / box * grid).astype(cp.int32)
            idx = cp.clip(idx, 0, grid - 1)

            rho_grid = cp.zeros((grid, grid, grid), dtype=cp.float64)
            cp.add.at(rho_grid, (idx[:,0], idx[:,1], idx[:,2]), charges_gpu)

            # 2. FFT to reciprocal space
            rho_k = fftn(rho_grid)

            # 3. Compute electric field in reciprocal space
            E_k_x = -1j * kx * prefactor * rho_k * exp_term
            E_k_y = -1j * ky * prefactor * rho_k * exp_term
            E_k_z = -1j * kz * prefactor * rho_k * exp_term

            # 4. Inverse FFT to real-space electric field
            E_grid_x = cp.real(ifftn(E_k_x))
            E_grid_y = cp.real(ifftn(E_k_y))
            E_grid_z = cp.real(ifftn(E_k_z))

            # 5. Interpolate back to atoms
            forces = cp.zeros_like(q_gpu)
            forces[:,0] = charges_gpu * E_grid_x[idx[:,0], idx[:,1], idx[:,2]]
            forces[:,1] = charges_gpu * E_grid_y[idx[:,0], idx[:,1], idx[:,2]]
            forces[:,2] = charges_gpu * E_grid_z[idx[:,0], idx[:,1], idx[:,2]]

            return forces.get()  # back to CPU as numpy array

        # Use GPU version if available
        compute_pme_reciprocal_forces = _compute_pme_reciprocal_forces_gpu
    else:
        # Fallback to CPU
        compute_pme_reciprocal_forces = _compute_pme_reciprocal_forces_cpu
else:
    compute_pme_reciprocal_forces = _compute_pme_reciprocal_forces_cpu

def compute_total_pme_forces(q, coulomb_pair_list, charges, C, box_size, cutoff):
    """Master PME function for forces."""
    alpha = 3.0 / cutoff 
    grid_size = 32
    f_real = compute_ewald_real_forces(q, coulomb_pair_list, charges, C, alpha, box_size, cutoff)
    f_recip = compute_pme_reciprocal_forces(q, charges, C, alpha, box_size, grid_size)
    return f_real + f_recip