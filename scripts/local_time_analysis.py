#scripts/local_time_analysis.py
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

q0, masses, charges, bond_list, angle_list, lj_pair_list, coulomb_pair_list = build_water_box(NUM_MOLECULES, spacing=3.1)
p0 = np.zeros_like(q0)

ff = SPCForceField(
    bond_list, angle_list, lj_pair_list, coulomb_pair_list, charges, 
    box_size=None, cutoff=None, mode=ELECTROSTATICS_MODE
)

# ==========================================
# 2. WARM UP PHASE (Get the atoms moving!)
# ==========================================
print("Warming up system so velocities are not zero...")
dt_warmup_akma = 0.1 / AKMA_TIME_TO_FS
q_warm, p_warm, _ = velocity_verlet(
    q0, p0, masses, force_func=ff.get_forces, dt=dt_warmup_akma, 
    num_steps=100, box_size=None, energy_func=ff.get_potential_energy, track_interval=100
)

# Use the state at step 100 as our new starting point
q_start = q_warm[-1]
p_start = p_warm[-1]

# ==========================================
# 3. RUN LOCAL ERROR EXPERIMENTS
# ==========================================
test_dts_fs = [0.01, 0.02, 0.05, 0.1, 0.2]
local_errors = []

dt_fine_fs = 0.0001
dt_fine_akma = dt_fine_fs / AKMA_TIME_TO_FS

for dt_fs in test_dts_fs:
    print(f"Measuring Local Error for dt = {dt_fs} fs...")
    dt_akma = dt_fs / AKMA_TIME_TO_FS
    
    # A. ONE single step from the WARM state
    q_approx, _, _ = velocity_verlet(
        q_start, p_start, masses, force_func=ff.get_forces, dt=dt_akma, 
        num_steps=1, box_size=None, energy_func=ff.get_potential_energy, track_interval=1
    )
    pos_approx = q_approx[-1] 
    
    # B. Many TINY steps from the WARM state
    num_fine_steps = int(round(dt_fs / dt_fine_fs))
    q_exact, _, _ = velocity_verlet(
        q_start, p_start, masses, force_func=ff.get_forces, dt=dt_fine_akma, 
        num_steps=num_fine_steps, box_size=None, energy_func=ff.get_potential_energy, 
        track_interval=num_fine_steps 
    )
    pos_exact = q_exact[-1]
    
    # C. Calculate Error
    diff = pos_approx - pos_exact
    error = np.sqrt(np.mean(diff**2))
    local_errors.append(error)

# ==========================================
# 3. PLOT LOG-LOG GRAPH
# ==========================================
plt.figure(figsize=(8, 6))

# Plot our actual simulation data (let's use purple so it looks distinct from the global error plot)
plt.plot(test_dts_fs, local_errors, marker='s', linestyle='-', color='purple', label='Simulation Local Error')

# Draw a reference line with a mathematical slope of 3 (O(Δt^3))
ref_line = [local_errors[0] * (dt / test_dts_fs[0])**3 for dt in test_dts_fs]
plt.plot(test_dts_fs, ref_line, marker='', linestyle='--', color='red', label='Theoretical O(Δt³) Slope')

plt.xscale('log')
plt.yscale('log')

plt.title('Velocity Verlet Local Truncation Error Analysis\n(Position Error After One Step)')
plt.xlabel('Time Step [dt] (fs) - Log Scale')
plt.ylabel('Position Error [Å] - Log Scale')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

plt.savefig("local_error_plot.pdf")
print("Saved plot to local_error_plot.pdf")
plt.show()