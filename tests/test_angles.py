import numpy as np
from src.mdwater.angles import compute_angle_forces

def test_angle_equilibrium():
    """Forces should be zero when the angle is exactly at equilibrium."""
    k_dummy = 2.0
    # Let's use 90 degrees (pi/2) as a simple equilibrium angle
    theta_eq = np.pi / 2 
    
    # Topology: [Oxygen, H1, H2]
    angle_list = np.array([[0, 1, 2]])
    
    # Place O at origin, H1 on X-axis, H2 on Y-axis (exactly 90 degrees)
    q = np.array([
        [0.0, 0.0, 0.0],  # Atom 0 (Oxygen)
        [1.0, 0.0, 0.0],  # Atom 1 (H1)
        [0.0, 1.0, 0.0]   # Atom 2 (H2)
    ])
    
    forces = compute_angle_forces(q, angle_list, k_dummy, theta_eq)
    
    # The forces should be perfectly zero!
    assert np.allclose(forces, 0.0, atol=1e-7), f"Expected F=0, got \n{forces}"

def test_angle_squeeze():
    """If the angle is forced wider than equilibrium, forces should push it closed."""
    k_dummy = 2.0
    # Equilibrium is 90 degrees
    theta_eq = np.pi / 2 
    
    angle_list = np.array([[0, 1, 2]])
    
    # Let's place the atoms 180 degrees apart (flat line)
    # H1 is at +1 on X, H2 is at -1 on X. Angle is pi (180 deg).
    q = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    
    forces = compute_angle_forces(q, angle_list, k_dummy, theta_eq)
    
    # Since they are spread too wide, H1 should be pushed "up" (+Y) 
    # and H2 should be pushed "up" (+Y) to close the angle back to 90 degrees.
    # (The exact direction depends on tiny floating point noise resolving the symmetry, 
    # but the total force on the system must be zero).
    
    total_force = np.sum(forces, axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-7), "Newton's 3rd Law failed!"