import numpy as np
import matplotlib.pyplot as plt

def plot_real_stiffness(xyz_filename, dt_fs=0.25):
    # 1. Read the XYZ file
    frames = []
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()
        
    # Figure out number of atoms from the first line
    n_atoms = int(lines[0].strip())
    lines_per_frame = n_atoms + 2
    
    # Extract coordinates for each frame
    for i in range(0, len(lines), lines_per_frame):
        if i + lines_per_frame > len(lines): break # End of file
        
        frame_coords = []
        # Skip the 2 header lines, read the atoms
        for j in range(2, lines_per_frame):
            parts = lines[i+j].split()
            # Assuming format: Element X Y Z
            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            frame_coords.append(coords)
        frames.append(frame_coords)
        
    trajectory = np.array(frames) # Shape: (n_frames, n_atoms, 3)
    time = np.arange(len(trajectory)) * dt_fs

    # 2. Calculate Distances
    # ASSUMPTION: Atom 0 is O1, Atom 1 is H1, Atom 3 is O2
    # Adjust these indices if your topology is ordered differently!
    idx_O1 = 0
    idx_H1 = 1
    idx_O2 = 3
    
    # Distance = sqrt( (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2 )
    r_OH = np.linalg.norm(trajectory[:, idx_O1, :] - trajectory[:, idx_H1, :], axis=1)
    r_OO = np.linalg.norm(trajectory[:, idx_O1, :] - trajectory[:, idx_O2, :], axis=1)

    # 3. Plot the Data
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    #fig.suptitle("Separation of Timescales in SPC Water", fontsize=14, fontweight='bold')

    # Fast Scale
    axs[0].plot(time, r_OH, '-', color='green', lw=1.5, alpha=0.8)
    axs[0].set_ylabel("O-H Distance (Å)")
    axs[0].set_title("Fast Timescale: Intramolecular Bond Vibration ($V_{bond}$)")
    axs[0].grid(True, alpha=0.3)

    # Slow Scale
    axs[1].plot(time, r_OO, 'r-', lw=2)
    axs[1].set_ylabel("O-O Distance (Å)")
    axs[1].set_title("Slow Timescale: Intermolecular H-Bond Formation ($V_{LJ} + V_{Coulomb}$)")
    axs[1].set_xlabel("Simulation Time (femtoseconds)")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("real_stiffness_plot.pdf")
    print("Saved plot to real_stiffness_plot.pdf")
    plt.show()

# Run the function (Replace with your actual filename!)
plot_real_stiffness("2_water_Hbond_vacuum.xyz", dt_fs=0.25)