# scripts/time_analysis.py

import numpy as np
import matplotlib.pyplot as plt

# Engine Imports
from mdwater.velocity_verlet import velocity_verlet
from mdwater.force_field import SPCForceField
from mdwater.topology import build_water_box

# ==========================================
# 1. SETUP
# ==========================================
NUM_MOLECULES = 2
ELECTROSTATICS_MODE = "VACUUM"
AKMA_TIME_TO_FS = 48.88

# CRITICAL: We must simulate the exact same amount of "real time" 
# for every test so the global error accumulation is comparable.
TOTAL_SIM_TIME_FS = 500.0  

q0, masses, charges, bond_list, angle_list, lj_pair_list, coulomb_pair_list = build_water_box(NUM_MOLECULES, spacing=3.1)
p0 = np.zeros_like(q0)

ff = SPCForceField(
    bond_list, angle_list, lj_pair_list, coulomb_pair_list, charges, 
    box_size=None, cutoff=None, mode=ELECTROSTATICS_MODE
)

# ==========================================
# 2. RUN EXPERIMENTS
# ==========================================
# We will test time steps from very small (0.05 fs) to large (1.0 fs)
test_dts_fs = [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0]
energy_errors = []

for dt_fs in test_dts_fs:
    print(f"Running dt = {dt_fs} fs...")
    dt_akma = dt_fs / AKMA_TIME_TO_FS
    
    # Calculate how many steps we need to reach exactly 500 fs total
    num_steps = int(TOTAL_SIM_TIME_FS / dt_fs) 
    
    _, _, energies = velocity_verlet(
        q0, p0, masses, force_func=ff.get_forces, dt=dt_akma, 
        num_steps=num_steps, box_size=None, energy_func=ff.get_potential_energy, track_interval=1
    )
    
    # Measure the "wiggle size" (Standard Deviation of Total Energy)
    energy_std = np.std(energies["total"])
    energy_errors.append(energy_std)

# ==========================================
# 3. PLOT LOG-LOG GRAPH
# ==========================================
plt.figure(figsize=(8, 6))

# Plot our actual simulation data
plt.plot(test_dts_fs, energy_errors, marker='o', linestyle='-', color='blue', label='Simulation Error')

# Draw a reference line with a mathematical slope of 2 (O(Δt^2))
# We anchor it to our first point so they line up nicely
ref_line = [energy_errors[0] * (dt / test_dts_fs[0])**2 for dt in test_dts_fs]
plt.plot(test_dts_fs, ref_line, marker='', linestyle='--', color='red', label='Theoretical O(Δt²) Slope')

# Convert axes to logarithmic scale!
plt.xscale('log')
plt.yscale('log')

plt.title('Velocity Verlet Global Truncation Error Analysis\n(Total Simulation Time: 500 fs)')
plt.xlabel('Time Step [dt] (fs) - Log Scale')
plt.ylabel('Total Energy Fluctuations [Std Dev] - Log Scale')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

plt.savefig("error_scaling_plot.pdf")
print("Saved plot to error_scaling_plot.pdf")
plt.show()