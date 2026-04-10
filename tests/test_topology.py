# tests/test_topology.py
import numpy as np
from src.mdwater.topology import build_water_box

def test_build_water_box():
    num_mols = 7
    q0, masses, charges, bond_list, angle_list, lj_pair_list, coulomb_pair_list = build_water_box(num_mols, spacing=2.8)
    
    # Check the shapes of the outputs
    assert q0.shape == (num_mols * 3, 3), f"Expected q0 shape {(num_mols * 3, 3)}, got {q0.shape}"
    assert masses.shape == (num_mols * 3,), f"Expected masses shape {(num_mols * 3,)}, got {masses.shape}"
    assert charges.shape == (num_mols * 3,), f"Expected charges shape {(num_mols * 3,)}, got {charges.shape}"
    
    # Check the number of bonds and angles
    assert len(bond_list) == num_mols * 2, f"Expected {num_mols * 2} bonds, got {len(bond_list)}"
    assert len(angle_list) == num_mols, f"Expected {num_mols} angles, got {len(angle_list)}"
    
    # Check the number of LJ and Coulomb pairs
    expected_lj_pairs = num_mols * (num_mols - 1) // 2
    expected_coulomb_pairs = num_mols * (num_mols - 1) // 2 * 9
    assert len(lj_pair_list) == expected_lj_pairs, f"Expected {expected_lj_pairs} LJ pairs, got {len(lj_pair_list)}"
    assert len(coulomb_pair_list) == expected_coulomb_pairs, f"Expected {expected_coulomb_pairs} Coulomb pairs, got {len(coulomb_pair_list)}"