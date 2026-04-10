import numpy as np
import matplotlib.pyplot as plt

# Engine Imports (Straight from your codebase!)
from mdwater.velocity_verlet import velocity_verlet
from mdwater.force_field import SPCForceField
from mdwater.topology import build_water_box

# ==========================================
# 1. SETUP THE TEST PARAMETERS
# ==========================================
NUM_MOLECULES = 2
ELECTROSTATICS_MODE = "VACUUM" # Kept as vacuum so it runs quickly for the test
spacing = 3.1
AKMA_TIME_TO_FS = 48.88

# We want to simulate the same amount of physical time for each test.
# Let's run 500 femtoseconds total to clearly see the drift.
TOTAL_SIMULATION_TIME_FS = 500.0 

# The different dt values we want to test (in femtoseconds)
# 0.25 is very safe, 1.0 is pushing it, 2.5 will likely drift/explode!
dt_test_values = [0.25, 1.0, 2.5] 

plt.figure(figsize=(10, 6))
print("Starting Energy Conservation Test...")

# ==========================================
# 2. RUN SIMULATION FOR EACH dt
# ==========================================
for target_step_fs in dt_test_values:
    print(f"--- Running for dt = {target_step_fs} fs ---")
    
    # Calculate integration parameters
    num_steps = int(TOTAL_SIMULATION_TIME_FS / target_step_fs)
    dt = target_step_fs / AKMA_TIME_TO_FS
    
    # We must rebuild the box every loop so they all start from the EXACT same state!
    q0, masses, charges, bond_list, angle_list, lj_pair_list, coulomb_pair_list = build_water_box(NUM_MOLECULES, spacing=spacing)
    p0 = np.zeros_like(q0)
    
    # Setup Force Field (Using None for box/cutoff since we are in VACUUM mode)
    ff = SPCForceField(
        bond_list, angle_list, lj_pair_list, coulomb_pair_list, charges, 
        box_size=None, cutoff=None, mode=ELECTROSTATICS_MODE
    )
    
    # Run Integrator
    traj_q, traj_p, energies = velocity_verlet(
        q0, p0, masses, 
        force_func=ff.get_forces, 
        dt=dt, 
        num_steps=num_steps, 
        box_size=None,
        energy_func=ff.get_potential_energy,
        track_interval=1 # Track every step for high-resolution plotting
    )
    
    # Extract data for plotting
    recorded_steps = energies["step"]
    total_energy = energies["total"]
    
    # Convert step number to physical time in femtoseconds for the x-axis
    time_fs = np.array(recorded_steps) * target_step_fs
    #time_fs = recorded_steps * target_step_fs
    
    # Plot this dt's total energy line
    plt.plot(time_fs, total_energy, label=f"dt = {target_step_fs} fs", linewidth=2)

# ==========================================
# 3. FINALIZE THE PLOT
# ==========================================
plt.title(f"Energy Conservation vs. Time Step (N={NUM_MOLECULES}, Direct Coulomb calculation)", fontsize=14)
plt.xlabel("Simulation Time (fs)", fontsize=12)
plt.ylabel("Total Energy (Kcal/mol)", fontsize=12)
plt.legend(fontsize=11, loc="best")
plt.ylim(2.7, 3.0) # Forces the plot to zoom out!
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig("energy_conservation_dt_comparison.png", dpi=300)
print("\nPlot saved as 'energy_conservation_dt_comparison.png'!")
plt.show()