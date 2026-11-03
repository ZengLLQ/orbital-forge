use approx::assert_relative_eq;
use orbital_forge::basis::build_sto3g_basis;
use orbital_forge::dft::{lda_correlation, lda_exchange};
use orbital_forge::molecules::h2_molecule;
use orbital_forge::scf::SCFRoot;
use orbital_forge::types::SCFConfig;

#[test]
fn test_lda_exchange_values() {
    let rho = 0.5;
    let (eps_x, v_x) = lda_exchange(rho);

    assert!(eps_x < 0.0);
    assert!(v_x < 0.0);
    // V_x = 4/3 * eps_x
    assert_relative_eq!(v_x, (4.0 / 3.0) * eps_x, epsilon = 1e-10);
}

#[test]
fn test_lda_correlation_values() {
    for &rho in &[0.001, 0.01, 0.1, 1.0, 10.0] {
        let (eps_c, v_c) = lda_correlation(rho);
        // Correlation energy must always be negative
        assert!(eps_c <= 0.0);
        assert!(v_c <= 0.0);
    }
}

#[test]
fn test_h2_dft_lda_calculation() {
    let mol = h2_molecule(Some(1.4));
    let basis = build_sto3g_basis(&mol);

    let config = SCFConfig {
        max_iter: 50,
        energy_threshold: 1e-6,
        density_threshold: 1e-5,
        use_diis: true,
        diis_max_space: 6,
        is_dft: true,
    };

    let scf = SCFRoot::new(mol, basis, config);
    let res = scf.solve();

    assert!(res.converged, "DFT-LDA must converge for H2");
    // DFT-LDA yields bound molecular energy for H2
    assert!(res.total_energy < -0.6 && res.total_energy > -1.5);
    assert!(res.homo_lumo_gap > 0.0);
}
