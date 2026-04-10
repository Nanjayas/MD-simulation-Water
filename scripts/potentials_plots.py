import numpy as np
import matplotlib.pyplot as plt

# --- Exact Parameters from force_field.py ---
# 1. Bonds
K_BOND = 1059.162  # kcal/mol/A^2
REQ = 1.012          # A

# 2. Angles
K_ANGLE = 75.9    # kcal/mol/rad^2
THETA_EQ_DEG = 113.24 
THETA_EQ_RAD = np.radians(THETA_EQ_DEG)

# 3. Lennard-Jones (Oxygen-Oxygen)
A_LJ = 630735.0    # kcal/mol * A^12
B_LJ = 626.13       # kcal/mol * A^6

# 4. Coulomb (Electrostatics)
C_COULOMB = 332.06375 # AKMA Coulomb constant
q_O = -0.82           # e
q_H = 0.41            # e

# --- Setup the 2x2 Plot ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("SPC flexible Water Model: Potential Energy Terms", fontsize=16, fontweight='bold')

# --- 1. Harmonic Bond Potential ---
# Center the plot exactly +/- 0.2 Å around REQ
r_bond = np.linspace(REQ - 0.2, REQ + 0.2, 100)
v_bond = 0.5 * K_BOND * (r_bond - REQ)**2

axs[0, 0].plot(r_bond, v_bond, 'b-', lw=2)
axs[0, 0].set_title("Intramolecular: Harmonic Bond (O-H)")
axs[0, 0].set_xlabel("O-H Distance (Å)")
axs[0, 0].set_ylabel("Energy (kcal/mol)")
axs[0, 0].axvline(REQ, color='k', linestyle='--', alpha=0.5, label=f"Equilibrium: {REQ} Å")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# --- 2. Harmonic Angle Potential ---
# Center the plot exactly +/- 20 degrees around THETA_EQ_DEG
theta_deg = np.linspace(THETA_EQ_DEG - 20, THETA_EQ_DEG + 20, 100)
theta_rad = np.radians(theta_deg)
v_angle = 0.5 * K_ANGLE * (theta_rad - THETA_EQ_RAD)**2

axs[0, 1].plot(theta_deg, v_angle, 'g-', lw=2)
axs[0, 1].set_title("Intramolecular: Harmonic Angle (H-O-H)")
axs[0, 1].set_xlabel("H-O-H Angle (Degrees)")
axs[0, 1].set_ylabel("Energy (kcal/mol)")
axs[0, 1].axvline(THETA_EQ_DEG, color='k', linestyle='--', alpha=0.5, label=f"Equilibrium: {THETA_EQ_DEG}°")
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# --- 3. Lennard-Jones Potential (Using A/B format) ---
r_lj = np.linspace(2.5, 8.0, 200)
v_lj = (A_LJ / r_lj**12) - (B_LJ / r_lj**6)

axs[1, 0].plot(r_lj, v_lj, 'r-', lw=2)
axs[1, 0].set_title("Intermolecular: Lennard-Jones (O-O)")
axs[1, 0].set_xlabel("O-O Distance (Å)")
axs[1, 0].set_ylabel("Energy (kcal/mol)")
axs[1, 0].axhline(0, color='k', linestyle='-', lw=1)
axs[1, 0].set_ylim(-0.25, 1.0) 
axs[1, 0].grid(True, alpha=0.3)

# --- 4. Coulomb Potential ---
r_c = np.linspace(1.5, 8.0, 200)
v_coulomb_attract = C_COULOMB * (q_O * q_H) / r_c  
v_coulomb_repel = C_COULOMB * (q_H * q_H) / r_c    

axs[1, 1].plot(r_c, v_coulomb_attract, 'm-', lw=2, label="O-H Attraction")
axs[1, 1].plot(r_c, v_coulomb_repel, 'c-', lw=2, label="H-H Repulsion")
axs[1, 1].set_title("Intermolecular: Coulomb Electrostatics")
axs[1, 1].set_xlabel("Interatomic Distance (Å)")
axs[1, 1].set_ylabel("Energy (kcal/mol)")
axs[1, 1].axhline(0, color='k', linestyle='-', lw=1)
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("potential_terms_grid_exact.pdf")
print("Saved exact plot to potential_terms_grid_exact.pdf")
plt.show()