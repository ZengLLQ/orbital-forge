//! Molecular properties analysis: Mulliken population charges, dipole moments, and HOMO-LUMO gaps.

use crate::types::{BasisSet, Mat, Molecule, Vec3};

/// Conversion factor from atomic units of dipole moment (e * Bohr) to Debye.
pub const AU_TO_DEBYE: f64 = 2.541_746_23;

/// Computes Mulliken atomic partial charges from the density matrix P and overlap matrix S:
/// q_A = Z_A - \sum_{\mu \in A} (P * S)_{\mu\mu}
#[allow(clippy::needless_range_loop)]
pub fn mulliken_population_analysis(
    molecule: &Molecule,
    basis: &BasisSet,
    p: &Mat,
    s: &Mat,
) -> Vec<f64> {
    let ps = p * s;
    let n_atoms = molecule.atoms.len();
    let mut charges = vec![0.0; n_atoms];

    for a in 0..n_atoms {
        charges[a] = molecule.atoms[a].atomic_number as f64;
    }

    for (mu, bf) in basis.functions.iter().enumerate() {
        let atom_idx = bf.atom_index;
        charges[atom_idx] -= ps[(mu, mu)];
    }

    charges
}

/// Computes the molecular permanent electric dipole moment vector and magnitude in Debye.
pub fn compute_dipole_moment(
    molecule: &Molecule,
    basis: &BasisSet,
    p: &Mat,
    s: &Mat,
) -> (Vec3, f64) {
    // 1. Nuclear contribution: mu_nuc = sum_A Z_A * R_A
    let mut mu_nuc = Vec3::zeros();
    for atom in &molecule.atoms {
        mu_nuc += atom.position * atom.atomic_number as f64;
    }

    // 2. Electronic contribution: mu_elec = -Tr(P * D)
    // Approximate dipole matrix D via center-of-charge approximation or exact overlap weighting:
    // <mu| r |nu> approx (R_mu + R_nu)/2 * S_{mu,nu}
    let n = basis.len();
    let mut mu_elec = Vec3::zeros();

    for i in 0..n {
        for j in 0..n {
            let center_mid = (basis.functions[i].center + basis.functions[j].center) * 0.5;
            let s_ij = s[(i, j)];
            let d_ij = center_mid * s_ij;

            mu_elec -= d_ij * p[(i, j)];
        }
    }

    let mu_total = mu_nuc + mu_elec;
    let magnitude_debye = mu_total.norm() * AU_TO_DEBYE;

    (mu_total, magnitude_debye)
}
