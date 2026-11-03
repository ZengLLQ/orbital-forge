//! Kohn-Sham Density Functional Theory (DFT) Engine with Local Density Approximation (LDA).

use crate::types::{BasisSet, Mat, Molecule, Vec3};
use std::f64::consts::PI;

/// Evaluates Dirac / Slater exchange energy density and potential:
/// eps_x(rho) = -3/4 * (3/pi)^(1/3) * rho^(1/3)
/// v_x(rho) = -(3/pi)^(1/3) * rho^(1/3)
pub fn lda_exchange(rho: f64) -> (f64, f64) {
    if rho < 1e-15 {
        return (0.0, 0.0);
    }
    let c_x = (3.0 / PI).powf(1.0 / 3.0);
    let rho_third = rho.powf(1.0 / 3.0);

    let eps_x = -0.75 * c_x * rho_third;
    let v_x = -c_x * rho_third;

    (eps_x, v_x)
}

/// Evaluates Chachiyo (2016) correlation energy density and potential for unpolarized LDA:
/// Simple, robust, analytic formula with accuracy comparable to VWN.
/// eps_c(r_s) = a * ln(1 + b/r_s + b/r_s^2)
pub fn lda_correlation(rho: f64) -> (f64, f64) {
    if rho < 1e-15 {
        return (0.0, 0.0);
    }
    // Wigner-Seitz radius r_s = (3 / (4 * pi * rho))^(1/3)
    let r_s = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);

    let a = -0.04742;
    let b = 21.7392;

    let inv_rs = 1.0 / r_s;
    let inv_rs2 = inv_rs * inv_rs;

    let arg = 1.0 + b * inv_rs + b * inv_rs2;
    let eps_c = a * arg.ln();

    // d(eps_c)/d(r_s) = a * (-b/r_s^2 - 2b/r_s^3) / arg
    let d_eps_d_rs = a * (-b * inv_rs2 - 2.0 * b * inv_rs2 * inv_rs) / arg;
    // v_c = eps_c - (r_s / 3) * d_eps_d_rs
    let v_c = eps_c - (r_s / 3.0) * d_eps_d_rs;

    (eps_c, v_c)
}

/// Numerical grid point with 3D coordinate and quadrature weight.
#[derive(Debug, Clone)]
pub struct GridPoint {
    pub coord: Vec3,
    pub weight: f64,
}

/// Generates a multicenter atomic quadrature grid for the molecule.
pub fn generate_molecular_grid(molecule: &Molecule) -> Vec<GridPoint> {
    let mut grid = Vec::new();

    // Radial Euler-Maclaurin shells + Angular Lebedev / Gauss-Legendre points
    let n_radial = 35;
    let n_theta = 12;
    let n_phi = 24;

    for atom in &molecule.atoms {
        let r_cov = 1.2; // Atomic covalent radius in Bohr approx

        for ir in 1..=n_radial {
            let x = ir as f64 / (n_radial as f64 + 1.0);
            let r = r_cov * x / (1.0 - x); // Radial coordinate
            let dr_dx = r_cov / (1.0 - x).powi(2);
            let w_rad = (1.0 / (n_radial as f64 + 1.0)) * dr_dx * r * r * 4.0 * PI;

            for it in 0..n_theta {
                let theta = (it as f64 + 0.5) * PI / n_theta as f64;
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();
                let w_theta = (PI / n_theta as f64) * sin_theta;

                for ip in 0..n_phi {
                    let phi = ip as f64 * 2.0 * PI / n_phi as f64;
                    let sin_phi = phi.sin();
                    let cos_phi = phi.cos();
                    let w_phi = 2.0 * PI / n_phi as f64;

                    let point = atom.position
                        + Vec3::new(
                            r * sin_theta * cos_phi,
                            r * sin_theta * sin_phi,
                            r * cos_theta,
                        );
                    let weight = w_rad * (w_theta / (4.0 * PI)) * (w_phi / (2.0 * PI));

                    grid.push(GridPoint {
                        coord: point,
                        weight,
                    });
                }
            }
        }
    }
    grid
}

/// Computes the exchange-correlation Fock contribution V_xc and total Exc energy:
#[allow(clippy::needless_range_loop)]
pub fn compute_exchange_correlation(
    basis: &BasisSet,
    density: &Mat,
    grid: &[GridPoint],
) -> (Mat, f64) {
    let n = basis.len();
    let mut v_xc = Mat::zeros(n, n);
    let mut e_xc = 0.0;

    // Buffer to evaluate basis functions at grid point
    let mut chi = vec![0.0; n];

    for pt in grid {
        // 1. Evaluate basis functions at point
        for i in 0..n {
            chi[i] = evaluate_basis_function(&basis.functions[i], pt.coord);
        }

        // 2. Compute electron density rho(r) = sum_{mu, nu} P_{mu, nu} chi_mu chi_nu
        let mut rho = 0.0;
        for i in 0..n {
            for j in 0..n {
                rho += density[(i, j)] * chi[i] * chi[j];
            }
        }
        rho = rho.max(0.0);

        if rho > 1e-15 {
            let (eps_x, v_x) = lda_exchange(rho);
            let (eps_c, v_c) = lda_correlation(rho);

            let eps_xc = eps_x + eps_c;
            let v_eff = v_x + v_c;

            let w = pt.weight;
            e_xc += w * rho * eps_xc;

            // V_xc_{mu, nu} += w * v_eff * chi_mu * chi_nu
            for i in 0..n {
                for j in 0..=i {
                    let val = w * v_eff * chi[i] * chi[j];
                    v_xc[(i, j)] += val;
                    if i != j {
                        v_xc[(j, i)] += val;
                    }
                }
            }
        }
    }

    (v_xc, e_xc)
}

fn evaluate_basis_function(bf: &crate::types::ContractedGaussian, r: Vec3) -> f64 {
    let dr = r - bf.center;
    let poly = dr.x.powi(bf.l.0 as i32) * dr.y.powi(bf.l.1 as i32) * dr.z.powi(bf.l.2 as i32);
    let r2 = dr.norm_squared();

    let mut val = 0.0;
    for prim in &bf.primitives {
        val += prim.coeff * prim.norm * (-prim.alpha * r2).exp();
    }
    poly * val
}
