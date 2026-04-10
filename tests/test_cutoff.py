import numpy as np

def test_cutoff_cliffs():
    r_inside = 8.99
    
    # 1. Lennard-Jones Constants (from your SPC water)
    A_LJ = 629400.0
    B_LJ = 625.5
    
    # 2. Coulomb Constants (attraction between Oxygen -0.82 and Hydrogen +0.41)
    # C * q1 * q2
    q_O = -0.82
    q_H = 0.41
    C_COULOMB = 332.06375
    coulomb_prefactor = C_COULOMB * q_O * q_H
    
    # --- Calculate Energy at 8.99 Å ---
    lj_energy = (A_LJ / r_inside**12) - (B_LJ / r_inside**6)
    coulomb_energy = coulomb_prefactor / r_inside
    
    print(f"--- AT THE 9.0 Å CUTOFF BOUNDARY ---")
    
    print(f"\n1. LENNARD-JONES:")
    print(f"   Energy at 8.99 Å: {lj_energy:.5f} kcal/mol")
    print(f"   Energy at 9.01 Å: 0.00000 kcal/mol (Cutoff applied!)")
    print(f"   The 'Cliff' jump: {abs(lj_energy - 0.0):.5f} kcal/mol")
    
    print(f"\n2. COULOMB:")
    print(f"   Energy at 8.99 Å: {coulomb_energy:.5f} kcal/mol")
    print(f"   Energy at 9.01 Å: 0.00000 kcal/mol (Cutoff applied!)")
    print(f"   The 'Cliff' jump: {abs(coulomb_energy - 0.0):.5f} kcal/mol")
    
    print("\nCONCLUSION:")
    print("If the jump is big, the atoms experience a massive, fake shockwave of force!")

if __name__ == "__main__":
    test_cutoff_cliffs()