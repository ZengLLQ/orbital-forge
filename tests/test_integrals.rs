use approx::assert_relative_eq;
use orbital_forge::basis::build_sto3g_basis;
use orbital_forge::integrals::{
    compute_eri_tensor, compute_kinetic_matrix, compute_nuclear_matrix, compute_overlap_matrix,
};
use orbital_forge::molecules::h2_molecule;

#[test]
fn test_matrix_symmetries() {
    let mol = h2_molecule(Some(1.4));
    let basis = build_sto3g_basis(&mol);

    let s = compute_overlap_matrix(&basis);
    let t = compute_kinetic_matrix(&basis);
    let v = compute_nuclear_matrix(&basis, &mol);

    let n = basis.len();

    // Symmetry assertions: M[i, j] == M[j, i]
    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(s[(i, j)], s[(j, i)], epsilon = 1e-12);
            assert_relative_eq!(t[(i, j)], t[(j, i)], epsilon = 1e-12);
            assert_relative_eq!(v[(i, j)], v[(j, i)], epsilon = 1e-12);
        }
    }

    // Overlap bounds: |S_{ij}| <= 1.0
    for i in 0..n {
        for j in 0..n {
            assert!(s[(i, j)].abs() <= 1.0 + 1e-12);
        }
    }

    // Kinetic energy must be strictly positive on diagonal
    for i in 0..n {
        assert!(t[(i, i)] > 0.0);
    }

    // Nuclear attraction must be strictly negative on diagonal
    for i in 0..n {
        assert!(v[(i, i)] < 0.0);
    }
}

#[test]
fn test_eri_permutational_symmetry() {
    let mol = h2_molecule(Some(1.4));
    let basis = build_sto3g_basis(&mol);
    let eri = compute_eri_tensor(&basis);

    let n = basis.len();

    // Verify 8-fold symmetry for ERIs: (ij|kl) == (ji|kl) == (ij|lk) == (kl|ij)
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let v1 = eri.get(i, j, k, l);
                    let v2 = eri.get(j, i, k, l);
                    let v3 = eri.get(i, j, l, k);
                    let v4 = eri.get(k, l, i, j);

                    assert_relative_eq!(v1, v2, epsilon = 1e-12);
                    assert_relative_eq!(v1, v3, epsilon = 1e-12);
                    assert_relative_eq!(v1, v4, epsilon = 1e-12);
                }
            }
        }
    }
}
