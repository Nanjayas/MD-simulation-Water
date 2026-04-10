# src/mdwater/velocity_verlet.py
# Velocity Verlet integrator for molecular dynamics
import numpy as np

# velocity-verlet integrator that works for arbitrary particle counts and potentials

def velocity_verlet(q, p, masses, force_func, dt, num_steps, box_size=None, energy_func=None, track_interval=0, constraints=None):
    """Integrate Hamiltonian mechanics using velocity Verlet.

    Parameters
    ----------
    q : ndarray, shape (N, D)
        Initial positions of N particles in D dimensions.
    p : ndarray, shape (N, D)
        Initial momenta of the particles.
    masses : ndarray, shape (N,)
        Mass of each particle.
    force_func : callable
        Function `f(q)` returning forces array of shape (N, D).
    dt : float
        Time step.
    num_steps : int
        Number of steps to integrate.
    box_size : float or array-like, optional
        If provided, applies periodic boundary conditions with this box size (assumes cubic box).
    energy_func : callable, optional 
        Function to track conservation of energy
    track_interval : int, optional
        If energy_func is provided, how often to compute and print energies
    
    Returns
    -------
    traj_q : ndarray, shape (num_steps+1, N, D)
        Positions at each time step.
    traj_p : ndarray, shape (num_steps+1, N, D)
        Momenta at each time step.
    """
    q = np.array(q, dtype=float)
    p = np.array(p, dtype=float)
    masses = np.array(masses, dtype=float)
    n, dim = q.shape

    traj_q = np.zeros((num_steps + 1, n, dim))
    traj_p = np.zeros_like(traj_q)

    traj_q[0] = q
    traj_p[0] = p

    # precompute mass column for vectorized operations
    mcol = masses[:, None]

    # Energy tracking dictionary
    energies = {"kinetic": [], "potential": [], "total": [], "step": []}

    f = force_func(q)
    for i in range(num_steps):

        # --- ENERGY TRACKING (Only on specified intervals) ---
        if track_interval > 0 and (i % track_interval == 0):
            ke = np.sum((p**2) / (2.0 * mcol))
            pe = energy_func(q) if energy_func else 0.0
            energies["kinetic"].append(ke)
            energies["potential"].append(pe)
            energies["total"].append(ke + pe)
            energies["step"].append(i)

        # 1. update positions
        q = q + (p / mcol) * dt + 0.5 * (f / mcol) * dt ** 2

        # PBC (Periodic Boundary Conditions)
        if box_size is not None:
            q = q % box_size

        # 2. half-step momentum
        p_half = p + 0.5 * f * dt
        # 3. compute new forces
        f = force_func(q)
        # 4. full-step momentum
        p = p_half + 0.5 * f * dt

        traj_q[i + 1] = q
        traj_p[i + 1] = p

    # Capture the very last step if requested
    if track_interval > 0 and (num_steps % track_interval == 0):
        ke = np.sum((p**2) / (2.0 * mcol))
        pe = energy_func(q) if energy_func else 0.0
        energies["kinetic"].append(ke)
        energies["potential"].append(pe)
        energies["total"].append(ke + pe)
        energies["step"].append(num_steps)

    return traj_q, traj_p, energies