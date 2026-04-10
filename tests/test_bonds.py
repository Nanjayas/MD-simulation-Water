import numpy as np

# We pretend these functions already exist in our src folder!
from src.mdwater.bonds import compute_bond_potential, compute_bond_forces

def test_bond_stretch():
    k_dummy = 2.0
    req_dummy = 1.0
    
    # We have two atoms, connected by one bond
    bond_list = np.array([[0, 1]]) 
    
    q = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    # Pass the new arguments!
    v = compute_bond_potential(q, bond_list, k_dummy, req_dummy)
    f = compute_bond_forces(q, bond_list, k_dummy, req_dummy)
    
    assert np.isclose(v, 1.0)
    assert f[0, 0] == 2.0
    assert f[1, 0] == -2.0