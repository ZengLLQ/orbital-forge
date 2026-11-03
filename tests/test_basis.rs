use approx::assert_relative_eq;
use orbital_forge::basis::build_sto3g_basis;
use orbital_forge::integrals::compute_overlap_matrix;
use orbital_forge::molecules::{h2_molecule, water_molecule};

#[test]
fn test_h2_basis_structure() {
    let mol = h2_molecule(None);
    let basis = build_sto3g_basis(&mol);

    // H2 has 2 hydrogen atoms, 1 basis function each (1s) -> 2 functions total
    assert_eq!(basis.len(), 2);
    assert_eq!(basis.name, "STO-3G");

    for bf in &basis.functions {
        assert_eq!(bf.primitives.len(), 3); // 3 primitives in STO-3G
        assert_eq!(bf.angular_momentum(), 0); // s-orbital
    }
}

#[test]
fn test_water_basis_structure() {
    let mol = water_molecule();
    let basis = build_sto3g_basis(&mol);

    // Water: O (1s, 2s, 2px, 2py, 2pz) = 5 functions + 2 * H (1s) = 2 functions -> 7 total
    assert_eq!(basis.len(), 7);

    // Check angular momentum of oxygen orbitals:
    assert_eq!(basis.functions[0].angular_momentum(), 0); // O 1s
    assert_eq!(basis.functions[1].angular_momentum(), 0); // O 2s
    assert_eq!(basis.functions[2].angular_momentum(), 1); // O 2px
    assert_eq!(basis.functions[3].angular_momentum(), 1); // O 2py
    assert_eq!(basis.functions[4].angular_momentum(), 1); // O 2pz
    assert_eq!(basis.functions[5].angular_momentum(), 0); // H1 1s
    assert_eq!(basis.functions[6].angular_momentum(), 0); // H2 1s
}

#[test]
fn test_basis_self_overlap_normalization() {
    let mol = water_molecule();
    let basis = build_sto3g_basis(&mol);
    let s = compute_overlap_matrix(&basis);

    // Crucial Invariant: All diagonal elements of overlap matrix must be strictly 1.0 (normalized)!
    for i in 0..basis.len() {
        assert_relative_eq!(s[(i, i)], 1.0, epsilon = 1e-10);
    }
}
