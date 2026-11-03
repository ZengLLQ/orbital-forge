use orbital_forge::basis::build_sto3g_basis;
use orbital_forge::molecules::h2_molecule;
use orbital_forge::scf::SCFRoot;
use orbital_forge::types::SCFConfig;

#[test]
fn test_diis_acceleration_comparison() {
    let mol = h2_molecule(Some(1.4));
    let basis = build_sto3g_basis(&mol);

    // 1. With DIIS
    let config_diis = SCFConfig {
        max_iter: 50,
        energy_threshold: 1e-10,
        density_threshold: 1e-8,
        use_diis: true,
        diis_max_space: 6,
        is_dft: false,
    };
    let res_diis = SCFRoot::new(mol.clone(), basis.clone(), config_diis).solve();

    // 2. Without DIIS
    let config_no_diis = SCFConfig {
        max_iter: 50,
        energy_threshold: 1e-10,
        density_threshold: 1e-8,
        use_diis: false,
        diis_max_space: 6,
        is_dft: false,
    };
    let res_no_diis = SCFRoot::new(mol, basis, config_no_diis).solve();

    assert!(res_diis.converged);
    assert!(res_no_diis.converged);

    // Both should yield the same physical ground state energy within 1e-8
    assert!((res_diis.total_energy - res_no_diis.total_energy).abs() < 1e-8);
}
