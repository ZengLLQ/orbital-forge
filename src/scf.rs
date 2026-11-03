//! Self-Consistent Field (SCF) Roothaan-Hall and Kohn-Sham DFT solver.

use crate::dft::{compute_exchange_correlation, generate_molecular_grid};
use crate::diis::DIIS;
use crate::integrals::{
    compute_eri_tensor, compute_kinetic_matrix, compute_nuclear_matrix, compute_overlap_matrix,
    ERIContainer,
};
use crate::properties::{compute_dipole_moment, mulliken_population_analysis};
use crate::types::{BasisSet, Mat, Molecule, SCFConfig, SCFResult, HARTREE_TO_EV};
use nalgebra::SymmetricEigen;

pub struct SCFRoot {
    pub molecule: Molecule,
    pub basis: BasisSet,
    pub config: SCFConfig,
    pub s: Mat,
    pub s_half_inv: Mat,
    pub h_core: Mat,
    pub eri: ERIContainer,
}

impl SCFRoot {
    pub fn new(molecule: Molecule, basis: BasisSet, config: SCFConfig) -> Self {
        let s = compute_overlap_matrix(&basis);
        let s_half_inv = symmetric_orthogonalizer(&s);

        let t = compute_kinetic_matrix(&basis);
        let v = compute_nuclear_matrix(&basis, &molecule);
        let h_core = t + v;

        let eri = compute_eri_tensor(&basis);

        Self {
            molecule,
            basis,
            config,
            s,
            s_half_inv,
            h_core,
            eri,
        }
    }

    /// Solves the electronic Roothaan-Hall equations self-consistently.
    pub fn solve(&self) -> SCFResult {
        let n_basis = self.basis.len();
        let n_occ = self.molecule.num_occupied_orbitals();
        let e_nuc = self.molecule.nuclear_repulsion_energy();

        // 1. Initial Guess from Core Hamiltonian
        let mut p = self.initial_density_guess();

        let mut diis = DIIS::new(self.config.diis_max_space);
        let mut prev_energy = 0.0;
        let mut converged = false;
        let mut final_iter = 0;
        let mut orbital_energies = Vec::new();

        // Precompute DFT grid if in Kohn-Sham DFT mode
        let grid = if self.config.is_dft {
            generate_molecular_grid(&self.molecule)
        } else {
            Vec::new()
        };

        for iter in 1..=self.config.max_iter {
            final_iter = iter;

            // 2. Build Fock Matrix & Electronic Energy
            let (f, e_elec) = if self.config.is_dft {
                self.build_dft_fock(&p, &grid)
            } else {
                self.build_rhf_fock(&p)
            };

            let total_energy = e_elec + e_nuc;
            let delta_e = (total_energy - prev_energy).abs();

            // 3. DIIS Extrapolation
            let f_diis = if self.config.use_diis && iter > 1 {
                let err = DIIS::compute_error(&f, &p, &self.s, &self.s_half_inv);
                let (f_extrap, _) = diis.extrapolate(&f, &err);
                f_extrap
            } else {
                f.clone()
            };

            // 4. Transform Fock Matrix to Orthonormal Basis: F' = S^(-1/2) * F * S^(-1/2)
            let f_prime = &self.s_half_inv * &f_diis * &self.s_half_inv;

            // 5. Diagonalize F': F' * C' = C' * epsilon
            let eigen = SymmetricEigen::new(f_prime);
            let mut sorted_pairs: Vec<(f64, nalgebra::DVector<f64>)> = eigen
                .eigenvalues
                .iter()
                .zip(eigen.eigenvectors.column_iter())
                .map(|(&e, col)| (e, col.clone_owned()))
                .collect();
            sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // 6. Back-transform MO Coefficients: C = S^(-1/2) * C'
            let mut c_prime = Mat::zeros(n_basis, n_basis);
            orbital_energies.clear();
            for (idx, (energy, col)) in sorted_pairs.into_iter().enumerate() {
                orbital_energies.push(energy);
                c_prime.set_column(idx, &col);
            }
            let mo_coefficients = &self.s_half_inv * c_prime;

            // 7. Form New Density Matrix: P_{\mu\nu} = 2 * sum_{i=1}^{n_occ} C_{\mu i} C_{\nu i}
            let mut p_new = Mat::zeros(n_basis, n_basis);
            for i in 0..n_basis {
                for j in 0..n_basis {
                    let mut sum = 0.0;
                    for k in 0..n_occ {
                        sum += mo_coefficients[(i, k)] * mo_coefficients[(j, k)];
                    }
                    p_new[(i, j)] = 2.0 * sum;
                }
            }

            // 8. Check Convergence
            let delta_p = (&p_new - &p).norm();
            p = p_new;
            prev_energy = total_energy;

            if iter > 1
                && delta_e < self.config.energy_threshold
                && delta_p < self.config.density_threshold
            {
                converged = true;
                break;
            }
        }

        // 9. Post-SCF Properties Analysis
        let homo_idx = n_occ - 1;
        let lumo_idx = n_occ;
        let homo_e = orbital_energies[homo_idx];
        let lumo_e = orbital_energies[lumo_idx];
        let gap = (lumo_e - homo_e) * HARTREE_TO_EV;

        let mulliken = mulliken_population_analysis(&self.molecule, &self.basis, &p, &self.s);
        let (dipole_vec, dipole_debye) =
            compute_dipole_moment(&self.molecule, &self.basis, &p, &self.s);

        SCFResult {
            converged,
            iterations: final_iter,
            total_energy: prev_energy,
            electronic_energy: prev_energy - e_nuc,
            nuclear_repulsion_energy: e_nuc,
            orbital_energies,
            homo_index: homo_idx,
            lumo_index: lumo_idx,
            homo_energy: homo_e,
            lumo_energy: lumo_e,
            homo_lumo_gap: gap,
            mulliken_charges: mulliken,
            dipole_moment: [dipole_vec.x, dipole_vec.y, dipole_vec.z],
            dipole_magnitude_debye: dipole_debye,
        }
    }

    fn initial_density_guess(&self) -> Mat {
        let n = self.basis.len();
        let n_occ = self.molecule.num_occupied_orbitals();

        let h_prime = &self.s_half_inv * &self.h_core * &self.s_half_inv;
        let eigen = SymmetricEigen::new(h_prime);

        let mut sorted_pairs: Vec<(f64, nalgebra::DVector<f64>)> = eigen
            .eigenvalues
            .iter()
            .zip(eigen.eigenvectors.column_iter())
            .map(|(&e, col)| (e, col.clone_owned()))
            .collect();
        sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut c_prime = Mat::zeros(n, n);
        for (idx, (_, col)) in sorted_pairs.into_iter().enumerate() {
            c_prime.set_column(idx, &col);
        }
        let c = &self.s_half_inv * c_prime;

        let mut p = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n_occ {
                    sum += c[(i, k)] * c[(j, k)];
                }
                p[(i, j)] = 2.0 * sum;
            }
        }
        p
    }

    fn build_rhf_fock(&self, p: &Mat) -> (Mat, f64) {
        let n = self.basis.len();
        let mut f = self.h_core.clone();

        for i in 0..n {
            for j in 0..=i {
                let mut two_e = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        let coul = self.eri.get(i, j, k, l);
                        let exch = self.eri.get(i, k, j, l);
                        two_e += p[(k, l)] * (coul - 0.5 * exch);
                    }
                }
                f[(i, j)] += two_e;
                if i != j {
                    f[(j, i)] += two_e;
                }
            }
        }

        // Electronic Energy: E_elec = 1/2 * sum_{mu, nu} P_{mu, nu} * (H^core + F)_{mu, nu}
        let mut e_elec = 0.0;
        for i in 0..n {
            for j in 0..n {
                e_elec += 0.5 * p[(i, j)] * (self.h_core[(i, j)] + f[(i, j)]);
            }
        }

        (f, e_elec)
    }

    fn build_dft_fock(&self, p: &Mat, grid: &[crate::dft::GridPoint]) -> (Mat, f64) {
        let n = self.basis.len();
        let mut f = self.h_core.clone();

        // 1. Classical Coulomb Repulsion Matrix J: J_{mu, nu} = sum_{la, si} P_{la, si} (mu nu | la si)
        let mut j_mat = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut coul = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        coul += p[(k, l)] * self.eri.get(i, j, k, l);
                    }
                }
                j_mat[(i, j)] = coul;
                if i != j {
                    j_mat[(j, i)] = coul;
                }
            }
        }

        // 2. Exchange-Correlation Potential V_xc and Energy E_xc from numerical grid
        let (v_xc, e_xc) = compute_exchange_correlation(&self.basis, p, grid);

        // Fock matrix in Kohn-Sham: F = H^core + J + V_xc
        f += &j_mat + &v_xc;

        // DFT Electronic Energy: E_elec = Tr(P * H^core) + 1/2 Tr(P * J) + E_xc
        let mut e_hcore = 0.0;
        let mut e_coulomb = 0.0;
        for i in 0..n {
            for k in 0..n {
                e_hcore += p[(i, k)] * self.h_core[(i, k)];
                e_coulomb += 0.5 * p[(i, k)] * j_mat[(i, k)];
            }
        }
        let e_elec = e_hcore + e_coulomb + e_xc;

        (f, e_elec)
    }
}

/// Computes symmetric Löwdin orthogonalizer S^(-1/2) via symmetric diagonalization:
/// S = U * Lambda * U^T  ==>  S^(-1/2) = U * Lambda^(-1/2) * U^T
pub fn symmetric_orthogonalizer(s: &Mat) -> Mat {
    let eigen = SymmetricEigen::new(s.clone());
    let u = &eigen.eigenvectors;
    let n = s.nrows();

    let mut inv_sqrt_diag = Mat::zeros(n, n);
    for i in 0..n {
        let val = eigen.eigenvalues[i];
        assert!(
            val > 1e-10,
            "Linear dependency detected in basis set: eigenvalue = {}",
            val
        );
        inv_sqrt_diag[(i, i)] = 1.0 / val.sqrt();
    }

    u * inv_sqrt_diag * u.transpose()
}
