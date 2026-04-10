#src/mdwater/force_field.py
import numpy as np

from mdwater.bonds import compute_bond_forces, compute_bond_potential
from mdwater.angles import compute_angle_forces, compute_angle_potential
from mdwater.lennard_jones import compute_lj_forces, compute_lj_potential
from mdwater.coulomb import compute_coulomb_forces, compute_coulomb_potential
from mdwater.coulomb import compute_total_ewald_forces, compute_total_ewald_potential
from mdwater.coulomb import compute_total_pme_potential, compute_total_pme_forces

class SPCForceField:
    def __init__(self, bond_list, angle_list, lj_pair_list, coulomb_pair_list, charges, box_size=None, cutoff=None, mode="VACUUM"):
        """
        Initializes the SPC Force Field with the simulation's topology.
        """
        # If box_size and cutoff are None, we assume Vacuum Mode (no PBC, no Ewald nor PME)
        self.box_size = box_size
        self.cutoff = cutoff
        self.mode = mode.upper() # Store the mode (ensure uppercase)

        # Internal parameters for SPC water
        self.K_BOND = 1059.162  # kcal mol^-1 Å^-2
        self.K_ANGLE = 75.90    # kcal mol^-1 rad^-2
        # External parameters for SPC water
        self.A_LJ = 630735.0  # kcal mol^-1 Å^12
        self.B_LJ = 626.13  # kcal mol^-1 Å^6
        self.C_COULOMB = 332.0637  # kcal mol^-1 Å e^-2 (AKMA units)

        # SCP standard equilibrium values for few molecules in vacuum
        if self.mode == "VACUUM":
            self.REQ = 1.0  
            self.THETA_EQ = np.deg2rad(109.47)  
        # SPCflexible bulk water
        else:
            self.REQ = 1.012  
            self.THETA_EQ = np.deg2rad(113.24)

        
        # Topology and properties
        self.bond_list = bond_list
        self.angle_list = angle_list
        self.lj_pair_list = lj_pair_list
        self.coulomb_pair_list = coulomb_pair_list
        self.charges = charges

    def print_summary(self):
        """Prints a clean summary of the active simulation physics."""
        print("\n" + "="*45)
        print("    MDWATER SIMULATION CONDITIONS ")
        print("="*45)
        print(f" MODE:           {self.mode}")
        if self.mode == "VACUUM":
            print(" BOX SIZE:       Infinite")
            print(" CUTOFF:         None")
            print(" ELECTROSTATICS: Direct Coulomb")
        else:
            print(f" BOX SIZE:       {self.box_size} Å")
            print(f" CUTOFF:         {self.cutoff} Å")
            print(f" ELECTROSTATICS: {self.mode} Summation")
        print("="*45 + "\n")

    def get_forces(self, q):
        """Calculates and sums all forces (Internal + External)."""
        f_bonds = compute_bond_forces(q, self.bond_list, self.K_BOND, self.REQ, box_size=self.box_size)
        f_angles = compute_angle_forces(q, self.angle_list, self.K_ANGLE, self.THETA_EQ, box_size=self.box_size)
        f_lj = compute_lj_forces(q, self.lj_pair_list, self.A_LJ, self.B_LJ, box_size=self.box_size, cutoff=self.cutoff)
        
        # --- updated switch for adding PME mode ---
        if self.mode == "VACUUM":
            f_coulomb = compute_coulomb_forces(q, self.coulomb_pair_list, self.charges, self.C_COULOMB, box_size=None, cutoff=None)
        elif self.mode == "EWALD":
            f_coulomb = compute_total_ewald_forces(q, self.coulomb_pair_list, self.charges, self.C_COULOMB, box_size=self.box_size, cutoff=self.cutoff)
        elif self.mode == "PME":
            f_coulomb = compute_total_pme_forces(q, self.coulomb_pair_list, self.charges, self.C_COULOMB, box_size=self.box_size, cutoff=self.cutoff)
            
        return f_bonds + f_angles + f_lj + f_coulomb

    def get_potential_energy(self, q):
        """Calculates and sums all potential energies."""
        pe_bonds = compute_bond_potential(q, self.bond_list, self.K_BOND, self.REQ, box_size=self.box_size)
        pe_angles = compute_angle_potential(q, self.angle_list, self.K_ANGLE, self.THETA_EQ, box_size=self.box_size)
        pe_lj = compute_lj_potential(q, self.lj_pair_list, self.A_LJ, self.B_LJ, box_size=self.box_size, cutoff=self.cutoff)
        
        # --- THE NEW SWITCH ---
        if self.mode == "VACUUM":
            pe_coulomb = compute_coulomb_potential(q, self.coulomb_pair_list, self.charges, self.C_COULOMB, box_size=None, cutoff=None)
        elif self.mode == "EWALD":
            pe_coulomb = compute_total_ewald_potential(q, self.coulomb_pair_list, self.charges, self.C_COULOMB, box_size=self.box_size, cutoff=self.cutoff)
        elif self.mode == "PME":
            pe_coulomb = compute_total_pme_potential(q, self.coulomb_pair_list, self.charges, self.C_COULOMB, box_size=self.box_size, cutoff=self.cutoff)

        return pe_bonds + pe_angles + pe_lj + pe_coulomb