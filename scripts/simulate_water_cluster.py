# simulate_water_cluster.py
import numpy as np
#import io
#from contextlib import redirect_stdout
import matplotlib.pyplot as plt 

# Engine Imports
from mdwater.velocity_verlet import velocity_verlet
from mdwater.force_field import SPCForceField
from mdwater.topology import build_water_box

# Utils Imports
from mdwater.utils.exporting import export_xyz, write_simulation_log
from mdwater.utils.plotting import plot_energies, plot_oo_distance, plot_oh_distance

# ==========================================
# 1. INITIALIZE THE SIMULATION
# ==========================================
NUM_MOLECULES = 2 
SAVE_DATA = True # TO save the trajectory, plots energy and logs

# --- Electrostatics Switch ---
# Options: "VACUUM", "EWALD", "PME"
ELECTROSTATICS_MODE = "VACUUM"  # Change this to test different electrostatics modes (VACUUM, EWALD, PME)

spacing = 3.1 ## 3.1 #standard separation

# --- Time Step and Simulation Length ---
AKMA_TIME_TO_FS = 48.88
target_step_fs = 0.25 #0.25 #/0.01   
num_steps = 12000 #12000 #30000 # for rdf analysis       
dt = target_step_fs / AKMA_TIME_TO_FS  
TRACK_INTERVAL = 1 # Tracks energy every 10 steps to avoid tracking every single step
# ----------------------------------

q0, masses, charges, bond_list, angle_list, lj_pair_list, coulomb_pair_list = build_water_box(NUM_MOLECULES, spacing=spacing)

# --- 2-MOLECULE mode ------------------------
CREATE_HYDROGEN_BOND = True

if NUM_MOLECULES == 2:
    setup_name = "H-Bond Setup" if CREATE_HYDROGEN_BOND else "Repulsive Setup"
    TITLE_SUFFIX = f"(2 Molecules | Separation: {spacing} Å | {setup_name} | dt: {target_step_fs} fs)"
else:
    TITLE_SUFFIX = f"({NUM_MOLECULES} Water Molecules Cluster | dt: {target_step_fs} fs)"
# -----------------------------------------------
if CREATE_HYDROGEN_BOND and NUM_MOLECULES == 2:
    # Flip Molecule 2 so its Oxygen faces Molecule 1's Hydrogen
    O2_pos = q0[3].copy() 
    for i in range(3, 6):
        q0[i] = O2_pos - (q0[i] - O2_pos) 
# -----------------------------------------------

p0 = np.zeros_like(q0)
# We use random numbers scaled by the atomic masses to simulate heat
#p0 = np.random.normal(0.0, 1.0, size=q0.shape) 
#p0 = p0 * np.sqrt(masses[:, np.newaxis]) * 1.5 # 2.5 is a scaling factor for high heat

if ELECTROSTATICS_MODE == "VACUUM":
    current_box = None
    current_cutoff = None
    BASE_NAME = f"{NUM_MOLECULES}_water_vacuum"
else:
    # Ensure box is at least 20.0 Angstroms
    current_box = max(20.0, np.cbrt(30.0 * NUM_MOLECULES))
    current_cutoff = min(9.1, (current_box / 2.0) - 0.1)
    # --- TEMPORARY OVERRIDE FOR VMD VISUALIZATION ---
    # 1. Remove the 20.0 A minimum. Just use pure density volume!
    #current_box = np.cbrt(30.0 * NUM_MOLECULES) 
    # 2. Force the cutoff to be strictly L/2 (even if it's super small)
    #current_cutoff = (current_box / 2.0) - 0.01 
    BASE_NAME = f"{NUM_MOLECULES}_water_pbc_{ELECTROSTATICS_MODE}"
    print(f"VISUALIZATION MODE: Box={current_box:.2f}, Cutoff={current_cutoff:.2f}")
# ------------------------------------------------
    


# Initialize Force Field with the current electrostatics mode
ff = SPCForceField(
    bond_list, angle_list, lj_pair_list, coulomb_pair_list, charges, 
    box_size=current_box, cutoff=current_cutoff, mode=ELECTROSTATICS_MODE
)

ff.print_summary()

# ==========================================
# 2. RUN INTEGRATOR. OPTIONAL TRACKING
# ==========================================
print(f"Starting simulation for {num_steps} steps...")

traj_q, traj_p, energies = velocity_verlet(
    q0, p0, masses, 
    force_func=ff.get_forces, 
    dt=dt, 
    num_steps=num_steps, 
    box_size=current_box,
    energy_func=ff.get_potential_energy, # Pass the energy function
    track_interval=TRACK_INTERVAL        # Pass the interval
)
print("Simulation complete!")

# ==========================================
# 3. POST-PROCESSING & EXPORT
# ==========================================
if SAVE_DATA:

    # NEW: Extract the steps array
    recorded_steps = energies["step"]

    # Pass TITLE_SUFFIX instead of spacing
    plot_energies(recorded_steps, energies["kinetic"], energies["potential"], energies["total"], BASE_NAME, TITLE_SUFFIX, save_data=SAVE_DATA)
    
    if NUM_MOLECULES == 2:
        # We calculate the O-O distance using the trajectory
        oo_distances = np.linalg.norm(traj_q[:, 0] - traj_q[:, 3], axis=1)
        oo_distances_sampled = oo_distances[recorded_steps]
        
        # Pass TITLE_SUFFIX instead of spacing
        plot_oo_distance(recorded_steps, oo_distances_sampled, energies["potential"], BASE_NAME, TITLE_SUFFIX, save_data=SAVE_DATA)

        # --- NEW: Calculate and plot the O-H covalent bond distance ---
        # Atom 0 is Oxygen, Atom 1 is Hydrogen on the first molecule
        oh_distances = np.linalg.norm(traj_q[:, 0] - traj_q[:, 1], axis=1)
        oh_distances_sampled = oh_distances[recorded_steps]
        
        # Pass TITLE_SUFFIX
        plot_oh_distance(recorded_steps, oh_distances_sampled, BASE_NAME, TITLE_SUFFIX, save_data=SAVE_DATA)

    export_xyz(traj_q, NUM_MOLECULES, filename=f"{BASE_NAME}.xyz")
    # Save the Log file
    write_simulation_log(
        filename=f"{BASE_NAME}_log.txt", 
        mode=ELECTROSTATICS_MODE, 
        num_molecules=NUM_MOLECULES, 
        spacing=spacing, 
        num_steps=num_steps, 
        dt=dt, 
        ff=ff, 
        q0=q0, 
        p0=p0
    )
    # RDF Analysis: save .npy trajectory for analysis script
    #np.save("traj_q.npy", traj_q)
    #print("Saved trajectory as numpy array for RDF analysis!")

plt.show()


