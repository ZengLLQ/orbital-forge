use approx::assert_relative_eq;
use orbital_forge::basis::build_sto3g_basis;
use orbital_forge::molecules::{h2_molecule, heh_plus};
use orbital_forge::scf::SCFRoot;
use orbital_forge::types::SCFConfig;

#[test]
fn test_h2_rhf_ground_state() {
    let r_bohr = 1.40107;
    let mol = h2_molecule(Some(r_bohr));
    let basis = build_sto3g_basis(&mol);

    let config = SCFConfig {
        max_iter: 50,
        energy_threshold: 1e-10,
        density_threshold: 1e-8,
        use_diis: true,
        diis_max_space: 6,
        is_dft: false,
    };

    let scf = SCFRoot::new(mol, basis, config);
    let res = scf.solve();

    assert!(res.converged, "SCF must converge for H2");
    assert!(
        res.iterations < 15,
        "DIIS should converge H2 in under 15 iterations"
    );

    // Theoretical STO-3G H2 energy at 1.401 Bohr is -1.1167 Hartree
    assert_relative_eq!(res.total_energy, -1.1167, epsilon = 0.002);

    // Nuclear repulsion E_nuc = 1.0 / 1.40107 = 0.71374 Hartree
    assert_relative_eq!(res.nuclear_repulsion_energy, 1.0 / r_bohr, epsilon = 1e-5);

    // HOMO energy should be around -0.578 Hartree
    assert_relative_eq!(res.homo_energy, -0.578, epsilon = 0.02);

    // Homonuclear diatomic has identically zero dipole moment
    assert_relative_eq!(res.dipole_magnitude_debye, 0.0, epsilon = 1e-6);

    // Both hydrogen atoms must have zero net partial charge
    assert_relative_eq!(res.mulliken_charges[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(res.mulliken_charges[1], 0.0, epsilon = 1e-6);
}

#[test]
fn test_heh_plus_rhf_ground_state() {
    let r_bohr = 1.4632;
    let mol = heh_plus(Some(r_bohr));
    let basis = build_sto3g_basis(&mol);

    let config = SCFConfig {
        max_iter: 50,
        energy_threshold: 1e-10,
        density_threshold: 1e-8,
        use_diis: true,
        diis_max_space: 6,
        is_dft: false,
    };

    let scf = SCFRoot::new(mol, basis, config);
    let res = scf.solve();

    assert!(res.converged, "SCF must converge for HeH+");

    // STO-3G HeH+ total energy is approximately -2.844 Hartree
    assert_relative_eq!(res.total_energy, -2.844, epsilon = 0.02);

    // Heteronuclear cation must have nonzero permanent dipole moment
    assert!(res.dipole_magnitude_debye > 1.0);

    // Sum of Mulliken charges must equal net molecular charge (+1)
    let total_charge: f64 = res.mulliken_charges.iter().sum();
    assert_relative_eq!(total_charge, 1.0, epsilon = 1e-6);
}
