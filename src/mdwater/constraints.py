#src/mdwater/constraints.py

import numpy as np

class WaterConstraints:
    def __init__(self, bond_list, angle_list, req=1.000, theta_eq_rad=1.91061):
        """
        Initializes the constraints for rigid water.
        req: Equilibrium bond length (O-H)
        theta_eq_rad: Equilibrium angle in radians (109.47 degrees)
        """
        self.req = req
        # Use Law of Cosines to find the rigid H-H distance to lock the angle
        self.d_hh = np.sqrt(2 * req**2 * (1 - np.cos(theta_eq_rad)))
        
        # Build a master list of all constraints: (atom_i, atom_j, target_distance)
        self.constraints = []
        
        # 1. Add O-H bond constraints
        for (i, j) in bond_list:
            self.constraints.append((i, j, self.req))
            
        # 2. Add H-H angle constraints
        # Assuming angle_list contains tuples of (H1_idx, O_idx, H2_idx)
        for (h1, o, h2) in angle_list:
            self.constraints.append((h1, h2, self.d_hh))

    def apply_shake(self, q_old, q_new, masses, tol=1e-6, max_iter=100):
        """
        SHAKE algorithm: Iteratively corrects positions to satisfy distance constraints.
        """
        q = q_new.copy()
        w = 1.0 / masses.flatten() # Inverse masses (w_i = 1/m_i)
        
        for iteration in range(max_iter):
            max_error = 0.0
            
            for (i, j, target_d) in self.constraints:
                # Vector between unconstrained new positions
                q_ij = q[i] - q[j]
                # Vector between old positions (the axis we correct along)
                q_ij_old = q_old[i] - q_old[j]
                
                current_d2 = np.dot(q_ij, q_ij)
                target_d2 = target_d**2
                
                # Check how badly the constraint is broken
                error = abs(current_d2 - target_d2) / target_d2
                if error > max_error:
                    max_error = error
                    
                # Calculate the SHAKE geometric correction factor (g)
                # g = (d^2 - |q_ij|^2) / (2 * (w_i + w_j) * (q_ij_old dot q_ij))
                denominator = 2.0 * (w[i] + w[j]) * np.dot(q_ij_old, q_ij)
                
                # Prevent division by zero if atoms perfectly overlap (rare)
                if abs(denominator) < 1e-8:
                    continue 
                    
                g = (target_d2 - current_d2) / denominator
                
                # Push the atoms back into place based on their mass
                # Lighter atoms move more, heavier atoms (Oxygen) move less
                q[i] += w[i] * g * q_ij_old
                q[j] -= w[j] * g * q_ij_old
                
            # If all constraints are within tolerance, we are done!
            if max_error < tol:
                break
                
        return q

    def apply_rattle(self, q, p, masses, tol=1e-6, max_iter=100):
        """
        RATTLE algorithm: Iteratively corrects momenta so no velocity acts ALONG the bonds.
        """
        p_new = p.copy()
        w = 1.0 / masses.flatten()
        
        for iteration in range(max_iter):
            max_error = 0.0
            
            for (i, j, target_d) in self.constraints:
                q_ij = q[i] - q[j]
                
                # Calculate relative velocity: v_ij = (p_i * w_i) - (p_j * w_j)
                v_ij = (p_new[i] * w[i]) - (p_new[j] * w[j])
                
                # Check how much velocity is pointing along the bond
                # Ideally, v_ij dot q_ij should be exactly 0
                dot_prod = np.dot(v_ij, q_ij)
                
                if abs(dot_prod) > max_error:
                    max_error = abs(dot_prod)
                    
               # Calculate the RATTLE velocity correction factor (k)
                k = dot_prod / ((w[i] + w[j]) * target_d**2)
                
                # Remove that momentum from the atoms (Newton's Third Law)
                # Equal and opposite! No mass scaling for momentum.
                p_new[i] -= k * q_ij 
                p_new[j] += k * q_ij
                
            if max_error < tol:
                break
                
        return p_new