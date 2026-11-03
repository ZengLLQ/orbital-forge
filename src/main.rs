use clap::Parser;
use orbital_forge::{
    build_sto3g_basis, h2_molecule, heh_plus, linear_h3_plus, parse_xyz, water_molecule, Molecule,
    SCFConfig, SCFRoot, HARTREE_TO_EV,
};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "orbital-forge",
    author = "ZengLLQ",
    version,
    about = "Molecular Electronic Structure, Restricted Hartree-Fock (RHF) & Kohn-Sham DFT Engine"
)]
struct Cli {
    /// Built-in molecule target: h2, heh+, h2o, h3+
    #[arg(short, long, default_value = "h2")]
    molecule: String,

    /// Optional path to an external XYZ file
    #[arg(short, long)]
    xyz: Option<PathBuf>,

    /// Run Kohn-Sham Density Functional Theory (LDA) instead of Hartree-Fock
    #[arg(long, default_value_t = false)]
    dft: bool,

    /// Disable DIIS convergence acceleration
    #[arg(long, default_value_t = false)]
    no_diis: bool,

    /// Maximum SCF iterations
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Energy convergence threshold (Hartree)
    #[arg(long, default_value_t = 1e-10)]
    energy_thresh: f64,

    /// Output results as JSON
    #[arg(long, default_value_t = false)]
    json: bool,
}

fn main() {
    let args = Cli::parse();

    let mol: Molecule = if let Some(path) = args.xyz {
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read XYZ file {:?}: {}", path, e));
        parse_xyz("Custom Molecule", &content, 0, 1).expect("Failed to parse XYZ")
    } else {
        match args.molecule.to_lowercase().as_str() {
            "h2" => h2_molecule(None),
            "heh+" | "heh" => heh_plus(None),
            "h2o" | "water" => water_molecule(),
            "h3+" | "h3" => linear_h3_plus(),
            other => panic!(
                "Unknown built-in molecule: {}. Use h2, heh+, h2o, or h3+",
                other
            ),
        }
    };

    let basis = build_sto3g_basis(&mol);
    let config = SCFConfig {
        max_iter: args.max_iter,
        energy_threshold: args.energy_thresh,
        density_threshold: 1e-8,
        use_diis: !args.no_diis,
        diis_max_space: 8,
        is_dft: args.dft,
    };

    if !args.json {
        println!("===========================================================");
        println!("        OrbitalForge: Molecular Electronic Structure       ");
        println!("===========================================================");
        println!("Target Molecule : {}", mol.name);
        println!(
            "Method          : {}",
            if args.dft {
                "Kohn-Sham DFT (LDA)"
            } else {
                "Restricted Hartree-Fock (RHF)"
            }
        );
        println!("Basis Set       : STO-3G ({} basis functions)", basis.len());
        println!(
            "Electrons       : {} ({} occupied pairs)",
            mol.num_electrons(),
            mol.num_occupied_orbitals()
        );
        println!(
            "Nuclear Repuls. : {:.8} Hartree",
            mol.nuclear_repulsion_energy()
        );
        println!(
            "DIIS Acceleration: {}",
            if config.use_diis {
                "Enabled (dim=8)"
            } else {
                "Disabled"
            }
        );
        println!("-----------------------------------------------------------");
        println!("Starting Self-Consistent Field (SCF) Iterations...");
    }

    let scf = SCFRoot::new(mol.clone(), basis, config);
    let result = scf.solve();

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!(
            "\n>>> Convergence Status: {} in {} iterations <<<",
            if result.converged {
                "CONVERGED"
            } else {
                "TERMINATED"
            },
            result.iterations
        );
        println!(
            "Electronic Energy  : {:>16.10} Hartree",
            result.electronic_energy
        );
        println!(
            "Nuclear Repulsion  : {:>16.10} Hartree",
            result.nuclear_repulsion_energy
        );
        println!(
            "Total Energy       : {:>16.10} Hartree",
            result.total_energy
        );
        println!(
            "Total Energy (eV)  : {:>16.10} eV",
            result.total_energy * HARTREE_TO_EV
        );

        println!("\n--- Orbital Spectrum ---");
        println!(
            "HOMO (Orbital #{}) : {:>10.6} Hartree ({:>8.3} eV)",
            result.homo_index + 1,
            result.homo_energy,
            result.homo_energy * HARTREE_TO_EV
        );
        println!(
            "LUMO (Orbital #{}) : {:>10.6} Hartree ({:>8.3} eV)",
            result.lumo_index + 1,
            result.lumo_energy,
            result.lumo_energy * HARTREE_TO_EV
        );
        println!(
            "HOMO-LUMO Gap       : {:>10.6} Hartree ({:>8.3} eV)",
            result.homo_lumo_gap / HARTREE_TO_EV,
            result.homo_lumo_gap
        );

        println!("\n--- Mulliken Population Charges ---");
        for (idx, atom) in mol.atoms.iter().enumerate() {
            println!(
                "Atom {:>2} ({:>2}) : Net Charge = {:>+8.4}",
                idx + 1,
                atom.symbol,
                result.mulliken_charges[idx]
            );
        }

        println!("\n--- Electric Dipole Moment ---");
        println!(
            "Dipole Vector (Debye) : [{:>8.4}, {:>8.4}, {:>8.4}]",
            result.dipole_moment[0] * orbital_forge::properties::AU_TO_DEBYE,
            result.dipole_moment[1] * orbital_forge::properties::AU_TO_DEBYE,
            result.dipole_moment[2] * orbital_forge::properties::AU_TO_DEBYE
        );
        println!(
            "Dipole Magnitude      : {:>8.4} Debye",
            result.dipole_magnitude_debye
        );
        println!("===========================================================\n");
    }
}
