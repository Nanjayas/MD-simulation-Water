<img style="float: right;" src="img/FUBerlinLogo.png" width="100">  

# Molecular Dynamics of Water
Course: Computational Sciences W25/26  
Institute: Freie Universität Berlin  
Lecturer: Sebastian Matera, Luca Donati 

## Description
A Python-based Molecular Dynamics model for simulating water molecules using the Hamiltonian energy functional for a flexible SPC water model. 

This project simulates both isolated water clusters in a vacuum and bulk water systems using Periodic Boundary Conditions (PBC). It tracks kinematic and potential energies, conserves total energy, and exports trajectory data for visualization. 
We implemented a cut off for Lennard-Jones interaction (short ranged) and we approximated the Coulomb interaction (far field) by Ewald Summation technique and improve it by Particle Mesh Ewald (PME).

## Important ⚠️
If you see empty folders/scripts, it's because we still have all the code on `develop` branch, please check it.

## Visuals
Two water molecules in a purely repulsive initial configuration (vacuum).

<img style="center;" src="anim/2water.gif" width="600"/> 
 
Two water molecules configuration position to form Hydrogen bonds. (vacuum)
<img style="center;" src="anim/2water_Hbond.gif" width="600"/>

A cluster of water molecules, using periodic boundary conditions (PBC), cutoff for Lennard Jones potential and standard Ewald Summation for Coulomb interaction.
<img style="center;" src="anim/clusterwater.gif" width="600"/>

## Installation
To run this project on your own machine, you will need Python 3.8+

1. **Clone the repository:**
   ```bash
   git clone https://git.imp.fu-berlin.de/itzeljem79/molecular-dynamics-of-water.git
   cd molecular-dynamics-of-water
   ```
2. **Install the package:**
   This command reads the `setup.py` and `requirements.txt` to automatically install all dependencies (`numpy`, `matplotlib`, etc.) and links the `src` folder.
   ```bash
   pip install -e .
   ```

## Repository Structure 

```text
├── src/               
│   ├── mdwater/       # The core code: physic system implemented
│       └── angles.py  # Internal potential/force
│       └── bonds.py   # Internal potential/force
│       └── coulomb.py  # External potential, also include Ewald and PME methods
│       └── lennard_jones.py  # External potential
│       └── force_field.py  # Merge all the force terms acting on the system
│       └── topology.py  # Generate initial positions of each atom for each molecule
│       └── velocity_verlet.py  # Integrator method (symplectic)
│   ├── utils/         # Helper functions (exporting data, logging, etc.)
├── notebooks/         # Intuition notes/Trial and error/ Visualisation plots
├── scripts/           # The actual scripts you run to start the simulation
│   └── simulate_water_cluster.py
├── tests/             # Continuous integration tests (for pytest)
├── archive/           # !!! into the corresponding teamate branch: other implementations
├── tutorials/  # Detailed explanations of usage
├── setup.py           # For Python package
├── requirements.txt   # Libraries used
└── README.md          
```
## Usage
To run the primary water cluster simulation, simply execute the main script:
```bash
python scripts/simulate_water_cluster.py
```
This will output a log file, an `.xyz` trajectory file (which you can open in VMD or Ovito), and generate Matplotlib graphs of the system's thermodynamics. You can change variables like `NUM_MOLECULES` and `ELECTROSTATICS_MODE` with options: `"VACUUM", "EWALD", "PME"` directly at the top of the script.  

For a very detailed tutorial of the script, explanation and some simulations examples, please see `tutorials/usage_tutorial.pdf`
## Support
If you encounter any bugs or mathematical anomalies, please open an Issue on our GitLab repository.

## Roadmap
To do:
- [ ] Integration of the SHAKE/RATTLE algorithm for rigid water constraints (OPtional).
- [ ] Parallelization using Numba/Dask for performance optimization (Optional).

## Contributing
We welcome contributions! Please ensure you create a new branch for your feature and open a Merge Request on GitLab. 
Before submitting, please run the test suite to ensure no physics are broken:
```bash
pytest
```

## Authors and acknowledgment
Discussions, material and results from different perspectives were provided to improve the implementation by all the team. 💅 

**Team Members:**

Yajie Zhang, Ningxin Wang, Nandana Jaya Sunil Kumar, Ayday Iskenderova, Itzel Jessica Martinez Marcelo

## License
This project is created for academic purposes. 

