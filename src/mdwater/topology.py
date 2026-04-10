# src/mdwater/topology.py
import numpy as np

# ==========================================
# 1. THE UNIVERSE BUILDER
# ==========================================
def build_water_box(num_mols, spacing=4.0):
    """
    Automatically builds positions and topology for N water molecules.
    Args:
        - um_mols (int): Number of water molecules to build.
        - spacing (float): Distance between molecules in the initial grid (default: 4.0 Å).
    Returns: A tuple containing:
        - q0 (np.ndarray): Initial positions of all atoms (shape: [num_atoms, 3]).
        - masses (np.ndarray): Masses of all atoms (shape: [num_atoms]).
        - charges (np.ndarray): Charges of all atoms (shape: [num_atoms]).
        - bond_list (np.ndarray): List of bonded atom pairs (shape: [num_bonds, 2]).
        - angle_list (np.ndarray): List of angle triplets (shape: [num_angles, 3]).
        - lj_pair_list (np.ndarray): List of atom pairs for Lennard-Jones interactions (shape: [num_lj_pairs, 2]).
        - coulomb_pair_list (np.ndarray): List of atom pairs for Coulomb interactions (shape: [num_coulomb_pairs, 2]). 
        """

    print(f"Building universe with {num_mols} water molecules...")
    
    REQ = 1.0
    THETA_EQ = np.deg2rad(109.47)
    mol_template = np.array([
        [0.0, 0.0, 0.0],
        [REQ, 0.0, 0.0],
        [REQ * np.cos(THETA_EQ), REQ * np.sin(THETA_EQ), 0.0]
    ])

    q0, masses, charges = [], [], []
    bond_list, angle_list, lj_pair_list, coulomb_pair_list = [], [], [], []
    
    # Calculate grid size (e.g., 10 molecules needs a 3x3x3 grid)
    side = int(np.ceil(num_mols ** (1.0/3.0)))
    mol_idx = 0
    
    # 1. Generate Coordinates, Internal Topology, and Properties
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if mol_idx >= num_mols: break
                
                # Shift molecule to its grid position
                shift = np.array([x, y, z]) * spacing
                q0.extend(mol_template + shift)
                
                # Atom indices for this molecule
                O_idx = mol_idx * 3
                H1_idx = O_idx + 1
                H2_idx = O_idx + 2
                
                bond_list.extend([[O_idx, H1_idx], [O_idx, H2_idx]])
                angle_list.append([O_idx, H1_idx, H2_idx])
                
                # Masses and charges of Oxygen and Hydrogens:
                masses.extend([15.999, 1.008, 1.008])
                charges.extend([-0.82, 0.41, 0.41])
                
                mol_idx += 1

    # 2. Generate External Pair Topology
    for i in range(num_mols):
        for j in range(i + 1, num_mols):
            # Lennard-Jones (Oxygen to Oxygen only)
            lj_pair_list.append([i*3, j*3])
            
            # Coulomb (Every atom in mol i interacts with every atom in mol j)
            for atom_i in [i*3, i*3+1, i*3+2]:
                for atom_j in [j*3, j*3+1, j*3+2]:
                    coulomb_pair_list.append([atom_i, atom_j])

    return (np.array(q0), np.array(masses), np.array(charges),
            np.array(bond_list), np.array(angle_list),
            np.array(lj_pair_list), np.array(coulomb_pair_list))