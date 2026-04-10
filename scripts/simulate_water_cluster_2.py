# -*- coding: utf-8 -*-
# simulate_water_cluster_2.py
import sys
import os
import time
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))
import numpy as np
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt 
import os   # <--- 用于创建目录

# ========== GPU ACCELERATION SWITCH ==========
USE_GPU = False   # Set to True to use GPU, False to use CPU

# 创建结果根目录和版本子目录
os.makedirs("results", exist_ok=True)
if USE_GPU:
    results_dir = "results/gpu"
else:
    results_dir = "results/cpu"
os.makedirs(results_dir, exist_ok=True)

# Import coulomb_2 module and set its GPU flag BEFORE other modules import it
import mdwater.coulomb_3 as coulomb
coulomb.USE_GPU = USE_GPU
if USE_GPU:
    print(" Using GPU-accelerated reciprocal space forces ")
else:
    print(" Using CPU version of reciprocal space forces")

# 将 force_field 模块中的 coulomb 引用替换为我们的 GPU/CPU 版本
import mdwater.force_field as ff_mod
ff_mod.coulomb = coulomb
# ==========================================

# Engine Imports (these will now use the modified coulomb module)
from mdwater.velocity_verlet import velocity_verlet
from mdwater.force_field import SPCForceField
from mdwater.topology import build_water_box

# Utils Imports
from mdwater.utils.exporting import export_xyz, write_simulation_log
from mdwater.utils.plotting import plot_energies, plot_oo_distance

# ==========================================
# 1. INITIALIZE THE SIMULATION
# ==========================================
NUM_MOLECULES = 5  # Set to 2 to test the hydrogen bond configuration and potential bowl; increase for larger clusters
SAVE_DATA = True # TO save the trajectory, plots energy and logs

# --- NEW: The Electrostatics Switch ---
# Options: "VACUUM", "EWALD", "PME"
ELECTROSTATICS_MODE = "EWALD" 

spacing = 3.1 #standard separation
q0, masses, charges, bond_list, angle_list, lj_pair_list, coulomb_pair_list = build_water_box(NUM_MOLECULES, spacing=spacing)

# --- 2-MOLECULE mode ---
CREATE_HYDROGEN_BOND = True  
if CREATE_HYDROGEN_BOND and NUM_MOLECULES == 2:
    # Flip Molecule 2 so its Oxygen faces Molecule 1's Hydrogen
    O2_pos = q0[3].copy() 
    for i in range(3, 6):
        q0[i] = O2_pos - (q0[i] - O2_pos) 
# ---------------------------

p0 = np.zeros_like(q0)

# Determine box size and base filename (without path)
if ELECTROSTATICS_MODE == "VACUUM":
    current_box = None
    current_cutoff = None
    base_name = f"{NUM_MOLECULES}_water_vacuum"
else:
    # Ensure box is at least 20.0 Angstroms
    current_box = max(20.0, np.cbrt(30.0 * NUM_MOLECULES))
    current_cutoff = min(9.0, (current_box / 2.0) - 0.1)
    base_name = f"{NUM_MOLECULES}_water_pbc_{ELECTROSTATICS_MODE}"

# 将结果保存到对应的子目录，并添加 _cpu 或 _gpu 后缀
if USE_GPU:
    BASE_NAME = os.path.join(results_dir, f"{base_name}_gpu")
else:
    BASE_NAME = os.path.join(results_dir, f"{base_name}_cpu")

# Initialize Force Field with the current electrostatics mode
ff = SPCForceField(
    bond_list, angle_list, lj_pair_list, coulomb_pair_list, charges, 
    box_size=current_box, cutoff=current_cutoff, mode=ELECTROSTATICS_MODE
)

ff.print_summary()

# ==========================================
# 2. RUN INTEGRATOR. OPTIONAL TRACKING
# ==========================================
AKMA_TIME_TO_FS = 48.88  
target_step_fs = 0.25    
num_steps = 12000        
dt = target_step_fs / AKMA_TIME_TO_FS  
TRACK_INTERVAL = 10 # Tracks energy every 10 steps

print(f"Starting simulation for {num_steps} steps...")
start_time = time.time()

traj_q, traj_p, energies = velocity_verlet(
    q0, p0, masses, 
    force_func=ff.get_forces, 
    dt=dt, 
    num_steps=num_steps, 
    box_size=current_box,
    energy_func=ff.get_potential_energy, # Pass the energy function
    track_interval=TRACK_INTERVAL        # Pass the interval
)
end_time = time.time()
elapsed = end_time - start_time
print(f"Simulation complete! Time taken: {elapsed:.2f} seconds")

time_filename = f"{BASE_NAME}_time.txt"
with open(time_filename, "w") as f:
    f.write(f"{elapsed:.2f}")

# ==========================================
# 3. POST-PROCESSING & EXPORT
# ==========================================
if SAVE_DATA:
    # 保存能量图（会自动加上 _time_plot.pdf 后缀）
    plot_energies(energies["kinetic"], energies["potential"], energies["total"], BASE_NAME, spacing, save_data=SAVE_DATA)
    
    if NUM_MOLECULES == 2:
        # 计算 O-O 距离并保存势阱图
        oo_distances = np.linalg.norm(traj_q[:, 0] - traj_q[:, 3], axis=1)
        oo_distances_sampled = oo_distances[energies["step"]]
        plot_oo_distance(oo_distances_sampled, energies["potential"], BASE_NAME, spacing, save_data=SAVE_DATA)

    # 保存轨迹
    export_xyz(traj_q, NUM_MOLECULES, filename=f"{BASE_NAME}.xyz")
    
    # 保存日志
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

# 不再调用 plt.show()，避免在服务器上卡住
# plt.show()