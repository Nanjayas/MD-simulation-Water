import numpy as np
import matplotlib.pyplot as plt

# --- Exact Parameters ---
A_LJ = 630735.0       # kcal/mol * A^12
B_LJ = 626.13         # kcal/mol * A^6
C_COULOMB = 332.06375 # AKMA Coulomb constant
q_O = -0.82           # e
q_H = 0.41            # e

# --- Setup the distance array (focusing on the tail end, from 3.5 to 15 Angstroms) ---
r = np.linspace(3.5, 15.0, 500)

# --- Calculate Energies (Absolute values for easy comparison) ---
# Lennard-Jones (O-O)
v_lj = np.abs((A_LJ / r**12) - (B_LJ / r**6))
# Coulomb (O-H attraction)
v_coulomb = np.abs(C_COULOMB * (q_O * q_H) / r)

# --- Calculate Forces (Absolute magnitude F = -dV/dr) ---
f_lj = np.abs((12 * A_LJ / r**13) - (6 * B_LJ / r**7))
f_coulomb = np.abs(C_COULOMB * (q_O * q_H) / r**2)

# --- Find exact values at 9.1 Angstroms ---
r_cut = 9
v_lj_cut = np.abs((A_LJ / r_cut**12) - (B_LJ / r_cut**6))
v_c_cut = np.abs(C_COULOMB * (q_O * q_H) / r_cut)

f_lj_cut = np.abs((12 * A_LJ / r_cut**13) - (6 * B_LJ / r_cut**7))
f_c_cut = np.abs(C_COULOMB * (q_O * q_H) / r_cut**2)

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#fig.suptitle("Why Cutoffs Work for Lennard-Jones but Fail for Coulomb", fontsize=15, fontweight='bold')

# 1. Energy Decay Plot
axs[0].plot(r, v_coulomb, 'm-', lw=2.5, label=r"Coulomb ($1/r$)")
axs[0].plot(r, v_lj, 'r-', lw=2.5, label=r"Lennard-Jones ($1/r^6$)")
axs[0].axvline(r_cut, color='k', linestyle='--', alpha=0.7, label=f"{r_cut} Å Cutoff")
axs[0].set_title("Potential Energy Decay Magnitude")
axs[0].set_xlabel("Distance (Å)", fontsize=11)
axs[0].set_ylabel("|Energy| (kcal/mol)", fontsize=11)
axs[0].set_ylim(-0.5, 15) # Zoom in to see the tail
axs[0].legend(fontsize=11)
axs[0].grid(True, alpha=0.3)

# Annotate the values at 9.1 A
axs[0].annotate(f'LJ Energy: ~{v_lj_cut:.3f}', xy=(r_cut, v_lj_cut), xytext=(r_cut+0.5, 2),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5))
axs[0].annotate(f'Coulomb Energy: ~{v_c_cut:.2f}', xy=(r_cut, v_c_cut), xytext=(r_cut+0.5, 12),
                arrowprops=dict(facecolor='magenta', shrink=0.05, width=1, headwidth=5))


# 2. Force Decay Plot
axs[1].plot(r, f_coulomb, 'm-', lw=2.5, label=r"Coulomb Force ($1/r^2$)")
axs[1].plot(r, f_lj, 'r-', lw=2.5, label=r"Lennard-Jones Force ($1/r^7$)")
axs[1].axvline(r_cut, color='k', linestyle='--', alpha=0.7, label=f"{r_cut} Å Cutoff")
axs[1].set_title("Force Decay Magnitude")
axs[1].set_xlabel("Distance (Å)", fontsize=11)
axs[1].set_ylabel("|Force| (kcal/mol/Å)", fontsize=11)
axs[1].set_ylim(-0.1, 5) # Zoom in to see the tail
axs[1].legend(fontsize=11)
axs[1].grid(True, alpha=0.3)

# Annotate the values at 9.1 A
axs[1].annotate(f'LJ Force: {f_lj_cut:.4f}', xy=(r_cut, f_lj_cut), xytext=(r_cut+0.5, 1.0),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5))
axs[1].annotate(f'Coulomb Force: ~{f_c_cut:.2f}', xy=(r_cut, f_c_cut), xytext=(r_cut+0.5, 3.5),
                arrowprops=dict(facecolor='magenta', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
plt.savefig("cutoff_comparison.pdf")
plt.show()