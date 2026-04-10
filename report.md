
1. **Introduction**

     Molecular dynamics (MD) is a computational technique used to simulate the time evolution of interacting particle systems by numerically solving Newton’s equations of motion. MD uses numerical integration technique to find the particle trajectories instead of estimating the analytical solutions for  many body systems which are generally impossible because of  non linear interactions.
     Water with its simple molecular structure  consisting of one oxygen and two hydrogen atoms shows complex physical behaviour due to electrostatic interactions and hydrogen bonding .this employed water as one of the crucial and extensively researched molecules to examine the molecular interactions and create effective simulation algorithms .
    In this project we conduct molecular dynamics simulation of water molecules ,characterised by a Hamiltonian that includes kinetic energy and potential energies indicating bond stretching ,angle bending ,Lennard-jones interactions and coulomb interactions. We use a second order accurate, symplectic integrator that holds energy conservation properties over long simulation times -velocity verlect method to solve the equations of motion.


2. **Mathematical Model**
             The system under study - water molecules moving in space characterised by Hamiltonian equation ,where the total energy is the sum of kinetic energy and potential energy contributions.
                     One water molecule contains 3 atoms consisting 1oxygen atom and 2 hydrogen atoms. Hence  N molecules contains 3N atoms and each atom has a position vector(q_i = (x_i, y_i, z_i),momentum vector {p}_i = (p_{x,i}, p_{y,i}, p_{z,i}),mass ( m_i),and forces acting on it({F}_i = (F_{x,i}, F_{y,i}, F_{z,i}).
      The total energy of the system is described by the Hamiltonian H = T + V(T- kinetic energy and V -potential energy). 
    Kinetic energy of n particles moves with momentum pi ∈R3 ,and having mass mᵢ
              T = 1/2 Σ (pᵢ² / mᵢ).
      Potential energy includes several contributions comes from the forces between atoms.
   1. Bond stretching :O-H bonds inside a water molecule behaves like springs hence it is represented using a harmonic spring potential V_{bond} = \frac{k}{2}(q - q_{eq})^2 .where q is the current bond length ,q_eq is the equilibrium bond length and K is the spring constant.
    2.Angle bending (H-O-H) : Water molecules have fixed bond angle(H — O — H104.5°),if this angle changes energy increases so V_{angle} = \frac{k_\theta}{2}(\theta - \theta_{eq})^2, where θ is the current angle and θ_eq is the equilibrium angle.
     3.Lennard-Jones Potential : Describes the short range interactions between molecules(Atoms very close → strong repulsion ,Atoms moderate distance → attraction).
          V_{LJ} =frac{A}{q^{12}} - \frac{B}{q^6}, where q-12 indicates strong repulsion and q-6 indicates attraction .This usually acts between oxygen atoms of different molecules.
     4.Coulomb Interaction : since the atoms in water molecules have partial charges they interact through electrostatic force,
         V_{Coul} =C \frac{C_i C_j}{q},where C_i ,C_j charges of atoms and q is the distance between them.
Hence  the total potential energy is the sum of these four interactions V =V_{bond}+V_{angle}+V_{LJ}+V_{Coul}.and the Hamiltonian becomes H = T + V_{bond} + V_{angle} + V_{LJ} + V_{Coul}.
         The motion of particles during simulation is governed by the forces ( using Newton’s law m_i \ddot{q_i} = F_i) comes from the derivative of these potentials(F = -\nabla V).

3.**Numerical Methods**
       The movement of atoms in a molecular system is governed by Newton’s equations of motion. For a particle i:

m_i \frac{d^2 q_i}{dt^2} = F_i

where
	•	m_i = mass of particle i
	•	q_i = position vector of the particle
	•	F_i = force acting on the particle

The force is obtained from the gradient of the potential energy:F_i = - \nabla V(r)
For systems with many interacting particles, these equations become nonlinear and coupled. So analytical solutions are generally impossible and numerical integration methods are used to approximate the evolution of the system over discretized small time step . In numerical simulations, time is divided into small intervals:

t, t + \Delta t, t + 2\Delta t, ...

where

\Delta t = simulation time step.

At each time step the algorithm updates:
	•	particle positions
	•	velocities
	•	forces
This allows us to compute the trajectory of the system over time.
  The simulation in this project uses the **velocity verlect method**, which is a widely used integrator in molecular dynamics simulations ,which conserves energy and is stable for long time simulations and it uses one force update per each time step.
1.position update: The position of the particle after \Delta t tilmestep q(t+\Delta t) =qt) + v(t)\Delta t + \frac{1}{2}a(t)\Delta t^2 , acceleration can be computed from Newton’s second law a(t) = \frac{F(t)}{m}
2.Computing New Force :Forces can be updated from the potential energy model (here bond stretching,lennard Jones interactions,Angle bending forces).this gives the updated acceleration a(t+\Delta t)).
3.Velocity update : The velocity is updated symmetrically using the average of updated and the previous accelerations, which improves the accuracy . v(t+\Delta t) = v(t)+1/2(a(t) + a(t+\Delta t))Delta t.
By repeating this process for many time steps we can simulate the motion of molecular system.

**Simulation Setup**
         We outline the water model geometry, force field parameters, initial conditions, periodic boundary conditions, and numerical integration strategy.The SPC (Simple Point Charge) model represents each water molecule as three point masses — one oxygen (O) and two hydrogen (H) atoms — connected by harmonic bonds with a fixed equilibrium geometry. Despite its simplicity, the SPC model reproduces many structural and thermodynamic properties of liquid water accurately.Each water molecule is constructed from a template placed at a grid point.
 The equilibrium geometry is defined by O-H bond length(r_eq)=1.0 Angstrom,H-O-H angle(theta_eq)=109.47 degrees, mass of Oxygen(m_O)015.999 amu, mass of Hydrogen(m_H)=1.008 amu ,charge of oxygen (q_O)=-0.82 elementary charge charge of hydrogen(q_H) =+0.41 elementary charge e.The two hydrogen atoms are placed symmetrically around the oxygen using the equilibrium bond length and angle. In the code, the molecule template is defined as: H1 = [r_eq, 0, 0],   H2 = [r_eq*cos(theta_eq), r_eq*sin(theta_eq), 0],   O = [0, 0, 0]

The total potential energy of the system consists of three contributions: internal bonded interactions (bonds and angles) and external non-bonded interactions (Lennard-Jones and Coulomb electrostatics). All parameters follow the standard SPC water model definition.
Internal (Bonded) Interactions
The intramolecular potential is modelled by harmonic springs for both the O-H bonds and the H-O-H bending angle: bond stiffness (k_bond)=1059.162 kcal/(mol*angstrom^2),angle stiffness (K_angle )=383.0 kcal/(mol*rad^2).
External (Non-bonded) Interactions
Intermolecular interactions are described by a Lennard-Jones (LJ) potential between oxygen atoms only, and Coulomb electrostatics between all atoms of different molecules:
LJ repulsion constant (A)=629400.0 kcal/(mol*Angstrom^12),LJ attraction constant (B)=625.5 kcal/(mol*Angstrom^6),Coulomb constant(C)=332.06375 kcal/(mol*Angstrom)/e^2.
The Lennard-Jones potential acts only between oxygen atoms (one per molecule), capturing van der Waals repulsion and dispersion attraction. Coulomb interactions act between every pair of charged atoms belonging to different molecules, giving 9 pairs per molecule pair (3 atoms x 3 atoms).

Simulation Block and Periodic Boundary Conditions
To simulate bulk liquid water rather than a finite cluster in vacuum, we employ Periodic Boundary Conditions (PBC). The simulation domain is a cubic box that tiles infinitely in all three spatial dimensions. When a molecule exits through one face of the box, it re-enters from the opposite face, eliminating surface effects.
Box parameters 
Number of molecules (N)=8(test/64 production),box edge length (L)=7.6(8mol)/12.4(64 mol) Angstrom,Molecule spacing (d) =3.8 Angstrom (grid spacing),cut off radius ( r_cut)<L/2 Angstrom (MIC requirement).
Minimum Image Convention(MIC)
With PBC, each atom interacts only with the nearest periodic image of every other atom. This is enforced by the Minimum Image Convention, applied to every pairwise distance vector:
 
dr = dr - L * round(dr / L)
 
This operation is applied in the bond, angle, Lennard-Jones, and Coulomb force routines. It ensures that the effective interaction distance never exceeds half the box length (L/2), which is the physical requirement for MIC to be valid.

Cutoff and Long -Range Electrostatics
Short-range Lennard-Jones forces decay rapidly (r^-12 and r^-6) and can safely be truncated at r_cut < L/2 with negligible error. A cutoff shift is applied so the potential reaches zero smoothly at the cutoff, avoiding energy discontinuities.
 
Coulomb interactions, however, decay only as r^-1 and cannot be truncated without significant error. The full treatment requires Ewald Summation, which splits the electrostatic sum into a short-range real-space contribution (handled with cutoff) and a long-range reciprocal-space contribution (handled via Fourier waves):
 
E_Coulomb = E_real + E_reciprocal - E_self
 
The parameter alpha controls the splitting between real and reciprocal space (alpha = 3.0 / r_cut), and k_max = 5 controls the number of reciprocal lattice vectors included in the summation.
Short-range Lennard-Jones forces decay rapidly (r^-12 and r^-6) and can safely be truncated at r_cut < L/2 with negligible error. A cutoff shift is applied so the potential reaches zero smoothly at the cutoff, avoiding energy discontinuities.
 
Coulomb interactions, however, decay only as r^-1 and cannot be truncated without significant error. The full treatment requires Ewald Summation, which splits the electrostatic sum into a short-range real-space contribution (handled with cutoff) and a long-range reciprocal-space contribution (handled via Fourier waves):
 
E_Coulomb = E_real + E_reciprocal - E_self
 
The parameter alpha controls the splitting between real and reciprocal space (alpha = 3.0 / r_cut), and k_max = 5 controls the number of reciprocal lattice vectors included in the summation.

Initial Conditions
Initial Positions:Molecules are placed on a regular cubic grid inside the simulation box. Each molecule is translated to its grid position using a spacing of 3.8 Angstrom, chosen to approximate the bulk water density of approximately 1 g/cm^3. The grid spacing determines the initial density:
 
rho = N * m_mol / L^3    (target: ~1 g/cm^3 for liquid water)
 Initial Momenta( Maxwell Boltzmann Distribution):Initial atomic momenta are sampled from the Maxwell-Boltzmann distribution at the target temperature T = 300 K. For each atom i, the momentum components are drawn from a Gaussian distribution with standard deviation:
 
sigma_i = sqrt(m_i * k_B * T)
 
where k_B = 0.001987 kcal/(mol*K) is the Boltzmann constant in AKMA units. After sampling, the centre-of-mass momentum is removed to prevent overall translation of the simulation box:
 
p = p - mean(p, axis=0)
Explicit initial Atom Positions(Single Molecule)
Every water molecule is built from the same template, then shifted to its grid position. The positions for one molecule (with oxygen at the origin) are ,first Hydrogen (H1) along X axis at r_eq =1Ang, the H2 x-coordinate is r_eq * cos(109.47 deg) = 1.0 * (-0.3333) = -0.3333 Ang, and the y-coordinate is r_eq * sin(109.47 deg) = 1.0 * 0.9428 = 0.9428 Ang. The molecule lies entirely in the x-y plane (z = 0).

Grid Positions for all 8 Molecules
The 8 molecules are placed on a 2 x 2 x 2 cubic grid with spacing 3.8 Angstrom. The oxygen atom of each molecule is placed at the following grid coordinates before adding the molecule template:
O1 (0,0,0),O2(0,01),O3 (0,1,0),O4(0,1,1),O5(1,0,0),O6(1,0,1),O7(1,1,0),O8(1,1,1). Each molecule's three atoms (O, H1, H2) are then placed by adding the molecule template to the grid shift vector. For example, molecule 1 has its oxygen at [0.0, 0.0, 3.8], H1 at [1.0, 0.0, 3.8], and H2 at [-0.3333, 0.9428, 3.8].
Initial Momenta : For the single molecule vacuum test, a small asymmetric nudge is applied to trigger visible motion. The symmetric opposing nudge on H1 (+0.8) and H2(-0.8) ensures zero total momentum (no centre-of-mass drift) while exciting both the bond stretching and angle bending modes simultaneously. For the multi-molecule PBC simulation, initial momenta are instead sampled from the Maxwell-Boltzmann distribution at T = 300 K.

 Numerical Integration : velocity Verlet 
Newton's equations of motion are integrated using the Velocity Verlet algorithm, which is a second-order symplectic integrator. It provides excellent long-term energy conservation and time-reversibility, making it the standard choice for molecular dynamics.
 
The algorithm advances the system by one time step dt as follows:
 
• Half-step momentum update:   p_(n+1/2) = p_n + (1/2) * F(q_n) * dt
• Full-step position update:   q_(n+1) = q_n + (p_(n+1/2) / m) * dt
• Apply PBC:                   q_(n+1) = q_(n+1) mod L
• Recompute forces:            F(q_(n+1))
• Complete momentum update:    p_(n+1) = p_(n+1/2) + (1/2) * F(q_(n+1)) * dt
A small time step (0.0002 AKMA) is required for the flexible SPC model because the O-H bond vibrations occur on a timescale of ~10 fs, requiring at least 10 time steps per oscillation for accurate integration. Larger time steps lead to energy drift and numerical instability.

Molecular Topology and Pair Lists 
The simulation requires pre-computed lists of interacting atom pairs to evaluate forces efficiently. These are constructed once during setup and reused at every time step.For 8 molecules (24 atoms), this results in 16 bonds( 2 per molecule - O-H1and O-H2), 8 angles(H1-O-H2), 28 LJ pairs(O-O) and 252 Coulomb pairs. The Coulomb pair count scales as 9*N*(N-1)/2 because each pair of molecules contributes 3*3 = 9 charged atom pairs.
The simulation supports two modes: Vacuum mode (single molecule, no PBC, simple Coulomb) for validation, and Bulk Liquid mode (multiple molecules, PBC active, Ewald summation) for physically meaningful liquid water simulation.
Before running the full PBC simulation, the implementation is validated using a single water molecule in vacuum with zero periodic boundary conditions. This test verifies:
 
• The Velocity Verlet integrator conserves total energy (drift < 0.0001 kcal/mol over 5000 steps)
• Bond lengths oscillate symmetrically around the equilibrium value r_eq = 1.0 Angstrom
• The H-O-H angle oscillates around the equilibrium value theta_eq = 109.47 degrees
• Forces satisfy Newton's third law (zero net force and torque on the molecule)
 
The single molecule test showed an energy drift of only 0.000012 kcal/mol over 5000 steps, confirming that the integrator and force field implementation are correct. This provides confidence in the multi-molecule PBC results.






 
 



            






                     