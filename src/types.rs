//! Fundamental data types and representations for molecular geometry and quantum states.

use nalgebra::{DMatrix, DVector, Vector3};
use serde::{Deserialize, Serialize};

/// Conversion factor from Ångströms to Bohr (atomic units)
pub const ANGSTROM_TO_BOHR: f64 = 1.889_726_124_565_062;
/// Conversion factor from Bohr to Ångströms
pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;
/// 1 Hartree in Electron-Volts (eV)
pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;

pub type Mat = DMatrix<f64>;
pub type VecN = DVector<f64>;
pub type Vec3 = Vector3<f64>;

/// Represents an atom in 3D space with atomic number and Cartesian coordinates in Bohr.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub symbol: String,
    pub atomic_number: usize,
    pub position: Vec3, // In Bohr (atomic units)
}

impl Atom {
    pub fn new(symbol: &str, atomic_number: usize, x: f64, y: f64, z: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            atomic_number,
            position: Vec3::new(x, y, z),
        }
    }

    /// Construct atom with coordinates specified in Ångströms (internally converted to Bohr).
    pub fn from_angstrom(
        symbol: &str,
        atomic_number: usize,
        x_ang: f64,
        y_ang: f64,
        z_ang: f64,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            atomic_number,
            position: Vec3::new(
                x_ang * ANGSTROM_TO_BOHR,
                y_ang * ANGSTROM_TO_BOHR,
                z_ang * ANGSTROM_TO_BOHR,
            ),
        }
    }
}

/// A molecular geometry composed of atoms with net charge and electron count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub name: String,
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: usize, // 1 for singlet (closed-shell RHF)
}

impl Molecule {
    pub fn new(name: &str, atoms: Vec<Atom>, charge: i32, multiplicity: usize) -> Self {
        Self {
            name: name.to_string(),
            atoms,
            charge,
            multiplicity,
        }
    }

    /// Total number of electrons in the molecule.
    pub fn num_electrons(&self) -> usize {
        let total_z: usize = self.atoms.iter().map(|a| a.atomic_number).sum();
        let n = total_z as i32 - self.charge;
        assert!(n > 0, "Molecule must have positive electron count");
        n as usize
    }

    /// Number of occupied spatial molecular orbitals in closed-shell RHF.
    pub fn num_occupied_orbitals(&self) -> usize {
        let n_elec = self.num_electrons();
        assert_eq!(
            n_elec % 2,
            0,
            "Closed-shell RHF requires even number of electrons"
        );
        n_elec / 2
    }

    /// Classical nuclear-nuclear Coulomb repulsion energy in Hartrees:
    /// E_nuc = sum_{A < B} (Z_A * Z_B) / |R_A - R_B|
    pub fn nuclear_repulsion_energy(&self) -> f64 {
        let mut e_nuc = 0.0;
        let n = self.atoms.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = (self.atoms[i].position - self.atoms[j].position).norm();
                if dist > 1e-10 {
                    e_nuc +=
                        (self.atoms[i].atomic_number * self.atoms[j].atomic_number) as f64 / dist;
                }
            }
        }
        e_nuc
    }
}

/// A primitive Cartesian Gaussian function:
/// g(r) = N * (x - Ax)^l * (y - Ay)^m * (z - Az)^n * exp(-alpha * |r - A|^2)
#[derive(Debug, Clone)]
pub struct PrimitiveGaussian {
    pub alpha: f64,
    pub coeff: f64,
    pub norm: f64,
    pub center: Vec3,
    pub l: (usize, usize, usize), // (l, m, n)
}

impl PrimitiveGaussian {
    pub fn angular_momentum(&self) -> usize {
        self.l.0 + self.l.1 + self.l.2
    }
}

/// A contracted Gaussian orbital (basis function) composed of normalized primitives:
/// chi(r) = sum_k d_k * g_k(r)
#[derive(Debug, Clone)]
pub struct ContractedGaussian {
    pub atom_index: usize,
    pub center: Vec3,
    pub l: (usize, usize, usize), // (lx, ly, lz)
    pub primitives: Vec<PrimitiveGaussian>,
}

impl ContractedGaussian {
    pub fn angular_momentum(&self) -> usize {
        self.l.0 + self.l.1 + self.l.2
    }
}

/// Molecular Basis Set composed of contracted Cartesian Gaussian functions.
#[derive(Debug, Clone)]
pub struct BasisSet {
    pub name: String,
    pub functions: Vec<ContractedGaussian>,
}

impl BasisSet {
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}

/// Configuration options for the Self-Consistent Field (SCF) solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCFConfig {
    pub max_iter: usize,
    pub energy_threshold: f64,
    pub density_threshold: f64,
    pub use_diis: bool,
    pub diis_max_space: usize,
    pub is_dft: bool,
}

impl Default for SCFConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            energy_threshold: 1e-10,
            density_threshold: 1e-8,
            use_diis: true,
            diis_max_space: 8,
            is_dft: false,
        }
    }
}

/// Output results from a converged (or terminated) SCF calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCFResult {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion_energy: f64,
    pub orbital_energies: Vec<f64>,
    pub homo_index: usize,
    pub lumo_index: usize,
    pub homo_energy: f64,
    pub lumo_energy: f64,
    pub homo_lumo_gap: f64,
    pub mulliken_charges: Vec<f64>,
    pub dipole_moment: [f64; 3],
    pub dipole_magnitude_debye: f64,
}
