#src/utils/exporting.py
import numpy as np
import io
from contextlib import redirect_stdout

def export_xyz(trajectory, num_mols, filename):
    """Exports the simulation trajectory to an XYZ file for VMD."""
    print(f"Exporting trajectory to {filename}...")
    num_frames = trajectory.shape[0]
    num_atoms = num_mols * 3
    atom_names = ["O", "H", "H"] * num_mols
    
    with open(filename, "w") as f:
        for step in range(num_frames):
            if step % 5 == 0:  # Save every 5th frame to save space
                f.write(f"{num_atoms}\n")
                f.write(f"Frame {step}\n")
                for i in range(num_atoms):
                    x, y, z = trajectory[step, i]
                    f.write(f"{atom_names[i]} {x:.5f} {y:.5f} {z:.5f}\n")
    print("Export complete!")

def write_simulation_log(filename, mode, num_molecules, spacing, num_steps, dt, ff, q0, p0):
    """Captures the force field summary and writes all initial conditions to a text file."""
    print(f"Writing initial conditions to {filename}...")
    
    # Catch the print_summary() output
    summary_catcher = io.StringIO()
    with redirect_stdout(summary_catcher):
        ff.print_summary() 
    ff_summary_text = summary_catcher.getvalue()

    # Write everything to the log file
    with open(filename, "w") as f:
        f.write("=== MDWATER SIMULATION LOG ===\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Number of Molecules: {num_molecules}\n")
        f.write(f"Separation: {spacing} Angstroms\n")
        f.write(f"Time steps: {num_steps}\n")
        f.write(f"Delta t: {dt:.5f} AKMA time units, {dt * 48.88:.2f} fs\n")
        f.write("================================\n")
        f.write(f"{ff_summary_text}\n")  
        f.write("================================\n")
        f.write(f"INITIAL POSITIONS (q0):\n{q0}\n\n")
        f.write(f"INITIAL MOMENTA (p0):\n{p0}\n")
    print("Log saved successfully!")