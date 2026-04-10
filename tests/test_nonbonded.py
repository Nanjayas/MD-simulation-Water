import numpy as np
from src.mdwater.nonbonded import compute_lj_forces, compute_lj_potential
from src.mdwater.nonbonded import compute_coulomb_forces, compute_coulomb_potential

def test_lj_forces_match_numerical():
    # SPC Water Constants
    A = 629400.0
    B = 625.5
    
    # Put two Oxygens exactly 3.0 Angstroms apart on the X-axis
    q = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    lj_pair_list = np.array([[0, 1]])
    
    # 1. Calculate the Analytical Force (Your Calculus)
    forces = compute_lj_forces(q, lj_pair_list, A, B)
    f_analytical = forces[0, 0]  # Force on atom 0 in the X direction
    
    # 2. Calculate the Numerical Force (central difference approximation)
    # We will nudge atom 0 slightly in the +X and -X direction and see how the potential changes.
    # F = V' ≈ -(V(q + delta) - V(q - delta)) / (2 * delta)
    delta = 1e-5
    
    # Nudge atom 0 forward
    q_plus = q.copy()
    q_plus[0, 0] += delta
    v_plus = compute_lj_potential(q_plus, lj_pair_list, A, B)
    
    # Nudge atom 0 backward
    q_minus = q.copy()
    q_minus[0, 0] -= delta
    v_minus = compute_lj_potential(q_minus, lj_pair_list, A, B)
    
    # Force is the NEGATIVE slope of the potential
    f_numerical = -(v_plus - v_minus) / (2.0 * delta)
    
    # 3. Prove they are mathematically identical!
    assert np.isclose(f_analytical, f_numerical, rtol=1e-4), \
        f"Mismatch! Analytical: {f_analytical}, Numerical: {f_numerical}"
    
def test_coulomb_forces_match_numerical():
    # SPC Coulomb Constant
    C_COULOMB = 332.06375
    
    # One Oxygen (-0.82) and one Hydrogen (+0.41)
    charges = np.array([-0.82, 0.41])
    
    # Put them exactly 2.0 Angstroms apart on the X-axis
    q = np.array([
        [0.0, 0.0, 0.0],  # Oxygen at origin
        [2.0, 0.0, 0.0]   # Hydrogen at x = 2.0
    ])
    
    coulomb_pair_list = np.array([[0, 1]])
    
    # 1. Calculate the Analytical Force (Your optimized math)
    forces = compute_coulomb_forces(q, coulomb_pair_list, charges, C_COULOMB)
    f_analytical = forces[0, 0]  # Force on the Oxygen in the X direction
    
    # 2. Calculate the Numerical Force (Finite Difference)
    delta = 1e-5
    
    # Nudge Oxygen forward
    q_plus = q.copy()
    q_plus[0, 0] += delta
    v_plus = compute_coulomb_potential(q_plus, coulomb_pair_list, charges, C_COULOMB)
    
    # Nudge Oxygen backward
    q_minus = q.copy()
    q_minus[0, 0] -= delta
    v_minus = compute_coulomb_potential(q_minus, coulomb_pair_list, charges, C_COULOMB)
    
    # Force is the NEGATIVE slope of the potential
    f_numerical = -(v_plus - v_minus) / (2.0 * delta)
    
    # 3. Prove they are mathematically identical!
    assert np.isclose(f_analytical, f_numerical, rtol=1e-4), \
        f"Mismatch! Analytical: {f_analytical}, Numerical: {f_numerical}"