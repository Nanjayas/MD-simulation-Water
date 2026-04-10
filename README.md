# Molecular Dynamics of Water

**Course:** Computational Sciences W25/26 — Freie Universität Berlin  
**Lecturers:** Sebastian Matera, Luca Donati

-----

## What is this project?

A Python-based Molecular Dynamics (MD) simulation engine for liquid water, built from scratch. We implemented the **flexible Simple Point Charge (SPC) water model**, which models each water molecule (H₂O) as three atoms with partial charges, flexible bonds, and a flexible angle.

The engine simulates both small vacuum systems and bulk water using Periodic Boundary Conditions (PBC).

-----

## What we simulated

- Two water molecules in vacuum (repulsive and hydrogen bond setups)
- Clusters of 20 and 50 water molecules in vacuum
- Bulk water with 27 molecules using Ewald Summation + PBC
- Bulk water with 512 molecules using Particle Mesh Ewald (PME)

-----

## Key Physics Implemented

|Component          |Description                                          |
|-------------------|-----------------------------------------------------|
|Hamiltonian        |Kinetic + bond + angle + Lennard-Jones + Coulomb     |
|Bond potential     |Harmonic O-H bond stretching                         |
|Angle potential    |Harmonic H-O-H angle bending                         |
|Lennard-Jones      |Short-range O-O interactions with spatial cutoff     |
|Coulomb            |Long-range electrostatics via Ewald Summation and PME|
|Integrator         |Velocity Verlet (symplectic, 2nd order)              |
|Boundary conditions|Periodic Boundary Conditions (PBC)                   |

-----

## What we validated

- ✅ Energy conservation in NVE ensemble
- ✅ Local and global truncation error analysis of Velocity Verlet (O(Δt³) and O(Δt²))
- ✅ Oxygen-Oxygen Radial Distribution Function (RDF) — first peak at ~2.8 Å with PME
- ✅ Velocity Autocorrelation Function (VACF) and simulated IR vibrational spectrum
- ✅ Symplectic (phase space volume preserving) properties of the integrator
- ✅ Rigid water model with SHAKE constraints (optional task)

-----

## Repository Structure

```
├── src/
│   ├── mdwater/
│       ├── angles.py          # Angle potential and force
│       ├── bonds.py           # Bond potential and force
│       ├── coulomb.py         # Coulomb + Ewald + PME
│       ├── lennard_jones.py   # Lennard-Jones potential
│       ├── force_field.py     # Combines all forces
│       ├── topology.py        # Initial positions
│       └── velocity_verlet.py # Integrator
│   └── utils/                 # Helper functions
├── notebooks/                 # Visualisation and analysis
├── scripts/
│   └── simulate_water_cluster.py
├── tests/                     # Unit tests (pytest)
├── tutorials/                 # Usage guide
├── setup.py
├── requirements.txt
└── README.md
```

-----

## Installation

Requires Python 3.8+

```bash
git clone https://github.com/Nanjayas/MD-simulation-Water.git
cd MD-simulation-Water
pip install -e .
```

-----

## Usage

```bash
python scripts/simulate_water_cluster.py
```

You can change these parameters at the top of the script:

- `NUM_MOLECULES` — number of water molecules
- `ELECTROSTATICS_MODE` — `"VACUUM"`, `"EWALD"`, or `"PME"`

Outputs: energy plots, RDF plot, `.xyz` trajectory file (view in VMD or Ovito)

-----

## Authors

Yajie Zhang · Ningxin Wang · **Nandana Jaya Sunil Kumar** · Ayday Iskenderova · Itzel Jessica Martinez Marcelo

*Group project — MSc Computational Sciences, Freie Universität Berlin, Winter Semester 2025/26*
