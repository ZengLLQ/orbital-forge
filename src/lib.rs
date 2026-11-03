//! # OrbitalForge
//!
//! High-Performance Molecular Electronic Structure, Restricted Hartree-Fock (RHF),
//! and Kohn-Sham Density Functional Theory (DFT) Engine in Rust.

pub mod basis;
pub mod dft;
pub mod diis;
pub mod integrals;
pub mod math;
pub mod molecules;
pub mod properties;
pub mod scf;
pub mod types;

pub use basis::build_sto3g_basis;
pub use dft::{compute_exchange_correlation, generate_molecular_grid};
pub use diis::DIIS;
pub use integrals::{
    compute_eri_tensor, compute_kinetic_matrix, compute_nuclear_matrix, compute_overlap_matrix,
};
pub use molecules::{h2_molecule, heh_plus, linear_h3_plus, parse_xyz, water_molecule};
pub use properties::{compute_dipole_moment, mulliken_population_analysis};
pub use scf::SCFRoot;
pub use types::{Atom, BasisSet, Molecule, SCFConfig, SCFResult, HARTREE_TO_EV};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
