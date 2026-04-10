# src/utils/plotting.py
import numpy as np
import matplotlib.pyplot as plt

def plot_energies(steps, kin_energies, pot_energies, tot_energies, base_name, title_suffix="", save_data=True):
    """Plots Kinetic, Potential, and Total Energy over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, pot_energies, label="Potential Energy", color="blue")
    plt.plot(steps, kin_energies, label="Kinetic Energy", color="orange")
    plt.plot(steps, tot_energies, label="Total Energy", color="black", linewidth=2)
    
    # NEW: Dynamic title
    plt.title(f"Energy Conservation vs Time\n{title_suffix}")
    plt.xlabel("Time (steps)")
    plt.ylabel("Energy (kcal/mol)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_data:
        plt.savefig(f"{base_name}_time_plot.pdf")
        print(f"Saved energy plot to {base_name}_time_plot.pdf")

def plot_oo_distance(steps, oo_distances, pot_energies, base_name, title_suffix="", save_data=True):
    """Plots the Intermolecular Potential Well (Energy vs Distance) with a time gradient."""
    plt.figure(figsize=(9, 6)) 
    
    time_steps = np.arange(len(oo_distances))
    scatter = plt.scatter(oo_distances, pot_energies, c=steps, cmap='viridis', s=5, alpha=0.8)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time (steps)')
    
    # NEW: Dynamic title and Å symbol
    plt.title(f"Intermolecular Potential Well (Energy vs Distance)\n{title_suffix}")
    plt.xlabel("O-O Distance (Å)")
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_data:
        plt.savefig(f"{base_name}_bowl_plot.pdf")
        print(f"Saved distance plot to {base_name}_bowl_plot.pdf")

def plot_oh_distance(steps, oh_distances, base_name, title_suffix="", save_data=True):
    """Plots the intramolecular O-H bond distance over time to visualize stiffness."""
    plt.figure(figsize=(10, 4))
    plt.plot(oh_distances, color='green', linewidth=1)
    
    # NEW: Å symbol
    # REQ = 1 for vacuum experiments, better for 2 molecules simulation 
    plt.axhline(y=1, color='red', linestyle='--', label="Ideal Bond Length (1 Å)")
    
    # NEW: Dynamic title
    plt.title(f"O-H Bond Distance vs Time (Vibrational Stiffness)\n{title_suffix}")
    plt.xlabel("Time (steps)")
    plt.ylabel("O-H Distance (Å)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_data:
        plt.savefig(f"{base_name}_oh_bond_plot.pdf")
        print(f"Saved O-H bond plot to {base_name}_oh_bond_plot.pdf")