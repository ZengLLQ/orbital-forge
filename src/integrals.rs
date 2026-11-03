//! Analytical evaluation of one-electron (Overlap, Kinetic, Nuclear Attraction)
//! and two-electron (Electron Repulsion) molecular integrals via Obara-Saika recurrence.

use crate::math::boys_function;
use crate::types::{BasisSet, ContractedGaussian, Mat, Molecule, PrimitiveGaussian, Vec3};
use std::f64::consts::PI;

/// Computes the symmetric N x N Overlap Matrix S:
/// S_{\mu\nu} = \int \chi_\mu(r) \chi_\nu(r) d^3r
pub fn compute_overlap_matrix(basis: &BasisSet) -> Mat {
    let n = basis.len();
    let mut s = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let val = contracted_overlap(&basis.functions[i], &basis.functions[j]);
            s[(i, j)] = val;
            s[(j, i)] = val;
        }
    }
    s
}

/// Computes the symmetric N x N Kinetic Energy Matrix T:
/// T_{\mu\nu} = \int \chi_\mu(r) (-1/2 \nabla^2) \chi_\nu(r) d^3r
pub fn compute_kinetic_matrix(basis: &BasisSet) -> Mat {
    let n = basis.len();
    let mut t = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let val = contracted_kinetic(&basis.functions[i], &basis.functions[j]);
            t[(i, j)] = val;
            t[(j, i)] = val;
        }
    }
    t
}

/// Computes the symmetric N x N Nuclear Attraction Matrix V:
/// V_{\mu\nu} = \sum_A \int \chi_\mu(r) (-Z_A / |r - R_A|) \chi_\nu(r) d^3r
pub fn compute_nuclear_matrix(basis: &BasisSet, mol: &Molecule) -> Mat {
    let n = basis.len();
    let mut v = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut val = 0.0;
            for atom in &mol.atoms {
                val += contracted_nuclear(
                    &basis.functions[i],
                    &basis.functions[j],
                    atom.position,
                    atom.atomic_number as f64,
                );
            }
            v[(i, j)] = val;
            v[(j, i)] = val;
        }
    }
    v
}

/// Storage structure for 4-center Two-Electron Repulsion Integrals (ERIs):
/// (\mu\nu|\lambda\sigma) = \int \int \chi_\mu(r1) \chi_\nu(r1) (1/r12) \chi_\lambda(r2) \chi_\sigma(r2) d3r1 d3r2
#[derive(Debug, Clone)]
pub struct ERIContainer {
    pub n_basis: usize,
    data: Vec<f64>,
}

impl ERIContainer {
    pub fn new(n_basis: usize) -> Self {
        let size = (n_basis * (n_basis + 1) / 2) * (n_basis * (n_basis + 1) / 2 + 1) / 2;
        Self {
            n_basis,
            data: vec![0.0; size.max(1)],
        }
    }

    #[inline]
    fn compound_index(i: usize, j: usize, k: usize, l: usize) -> usize {
        let (mu, nu) = if i >= j { (i, j) } else { (j, i) };
        let (la, si) = if k >= l { (k, l) } else { (l, k) };

        let ij = mu * (mu + 1) / 2 + nu;
        let kl = la * (la + 1) / 2 + si;

        let (p, q) = if ij >= kl { (ij, kl) } else { (kl, ij) };
        p * (p + 1) / 2 + q
    }

    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let idx = Self::compound_index(i, j, k, l);
        self.data[idx]
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
        let idx = Self::compound_index(i, j, k, l);
        self.data[idx] = val;
    }
}

/// Precomputes all 4-center Two-Electron Repulsion Integrals (ERIs) utilizing 8-fold symmetry.
pub fn compute_eri_tensor(basis: &BasisSet) -> ERIContainer {
    let n = basis.len();
    let mut eri = ERIContainer::new(n);

    for i in 0..n {
        for j in 0..=i {
            let ij = i * (i + 1) / 2 + j;
            for k in 0..n {
                for l in 0..=k {
                    let kl = k * (k + 1) / 2 + l;
                    if ij >= kl {
                        let val = contracted_eri(
                            &basis.functions[i],
                            &basis.functions[j],
                            &basis.functions[k],
                            &basis.functions[l],
                        );
                        eri.set(i, j, k, l, val);
                    }
                }
            }
        }
    }
    eri
}

// -----------------------------------------------------------------------------
// Analytical Primitive Integral Evaluations
// -----------------------------------------------------------------------------

fn contracted_overlap(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let mut sum = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            sum += pa.coeff * pb.coeff * pa.norm * pb.norm * primitive_overlap(pa, pb);
        }
    }
    sum
}

fn primitive_overlap(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> f64 {
    let gamma = a.alpha + b.alpha;
    let kab = (-a.alpha * b.alpha / gamma * (a.center - b.center).norm_squared()).exp();
    let px = (a.alpha * a.center.x + b.alpha * b.center.x) / gamma;
    let py = (a.alpha * a.center.y + b.alpha * b.center.y) / gamma;
    let pz = (a.alpha * a.center.z + b.alpha * b.center.z) / gamma;

    let sx = os_overlap_1d(a.l.0, b.l.0, a.center.x, b.center.x, px, gamma);
    let sy = os_overlap_1d(a.l.1, b.l.1, a.center.y, b.center.y, py, gamma);
    let sz = os_overlap_1d(a.l.2, b.l.2, a.center.z, b.center.z, pz, gamma);

    kab * sx * sy * sz
}

fn os_overlap_1d(l1: usize, l2: usize, ax: f64, bx: f64, px: f64, gamma: f64) -> f64 {
    let s00 = (PI / gamma).sqrt();
    if l1 == 0 && l2 == 0 {
        return s00;
    }
    if l1 == 1 && l2 == 0 {
        return (px - ax) * s00;
    }
    if l1 == 0 && l2 == 1 {
        return (px - bx) * s00;
    }
    if l1 == 1 && l2 == 1 {
        return (px - ax) * (px - bx) * s00 + 0.5 / gamma * s00;
    }
    if l1 == 2 && l2 == 0 {
        return (px - ax) * (px - ax) * s00 + 0.5 / gamma * s00;
    }
    if l1 == 0 && l2 == 2 {
        return (px - bx) * (px - bx) * s00 + 0.5 / gamma * s00;
    }
    // General Obara-Saika recurrence if higher angular momentum
    let mut table = vec![vec![0.0; l2 + 1]; l1 + 1];
    table[0][0] = s00;
    for i in 0..l1 {
        table[i + 1][0] = (px - ax) * table[i][0]
            + if i > 0 {
                (i as f64) / (2.0 * gamma) * table[i - 1][0]
            } else {
                0.0
            };
    }
    for j in 0..l2 {
        for i in 0..=l1 {
            table[i][j + 1] = (px - bx) * table[i][j]
                + if j > 0 {
                    (j as f64) / (2.0 * gamma) * table[i][j - 1]
                } else {
                    0.0
                }
                + if i > 0 {
                    (i as f64) / (2.0 * gamma) * table[i - 1][j]
                } else {
                    0.0
                };
        }
    }
    table[l1][l2]
}

fn contracted_kinetic(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let mut sum = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            sum += pa.coeff * pb.coeff * pa.norm * pb.norm * primitive_kinetic(pa, pb);
        }
    }
    sum
}

fn primitive_kinetic(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> f64 {
    let gamma = a.alpha + b.alpha;
    let kab = (-a.alpha * b.alpha / gamma * (a.center - b.center).norm_squared()).exp();
    let p = (a.alpha * a.center + b.alpha * b.center) / gamma;

    let sx = os_overlap_1d(a.l.0, b.l.0, a.center.x, b.center.x, p.x, gamma);
    let sy = os_overlap_1d(a.l.1, b.l.1, a.center.y, b.center.y, p.y, gamma);
    let sz = os_overlap_1d(a.l.2, b.l.2, a.center.z, b.center.z, p.z, gamma);

    let tx = os_kinetic_1d(
        a.l.0, b.l.0, a.alpha, b.alpha, a.center.x, b.center.x, p.x, gamma,
    );
    let ty = os_kinetic_1d(
        a.l.1, b.l.1, a.alpha, b.alpha, a.center.y, b.center.y, p.y, gamma,
    );
    let tz = os_kinetic_1d(
        a.l.2, b.l.2, a.alpha, b.alpha, a.center.z, b.center.z, p.z, gamma,
    );

    kab * (tx * sy * sz + sx * ty * sz + sx * sy * tz)
}

#[allow(clippy::too_many_arguments)]
fn os_kinetic_1d(
    l1: usize,
    l2: usize,
    _a: f64,
    b: f64,
    ax: f64,
    bx: f64,
    px: f64,
    gamma: f64,
) -> f64 {
    let s_plus2 = os_overlap_1d(l1, l2 + 2, ax, bx, px, gamma);
    let s_zero = os_overlap_1d(l1, l2, ax, bx, px, gamma);
    let s_minus2 = if l2 >= 2 {
        os_overlap_1d(l1, l2 - 2, ax, bx, px, gamma)
    } else {
        0.0
    };

    b * (2.0 * l2 as f64 + 1.0) * s_zero
        - 2.0 * b * b * s_plus2
        - 0.5 * (l2 as f64) * (l2 as f64 - 1.0) * s_minus2
}

fn contracted_nuclear(a: &ContractedGaussian, b: &ContractedGaussian, c: Vec3, z: f64) -> f64 {
    let mut sum = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            sum += pa.coeff * pb.coeff * pa.norm * pb.norm * primitive_nuclear(pa, pb, c, z);
        }
    }
    sum
}

#[allow(clippy::needless_range_loop)]
fn primitive_nuclear(a: &PrimitiveGaussian, b: &PrimitiveGaussian, c: Vec3, z: f64) -> f64 {
    let gamma = a.alpha + b.alpha;
    let kab = (-a.alpha * b.alpha / gamma * (a.center - b.center).norm_squared()).exp();
    let p = (a.alpha * a.center + b.alpha * b.center) / gamma;
    let r_pc2 = (p - c).norm_squared();
    let t = gamma * r_pc2;

    let max_m = a.l.0 + a.l.1 + a.l.2 + b.l.0 + b.l.1 + b.l.2;
    let mut f_vals = vec![0.0; max_m + 2];
    for m in 0..=max_m + 1 {
        f_vals[m] = boys_function(m, t);
    }

    let prefactor = -2.0 * PI / gamma * z * kab;

    // For s and p shells:
    if a.angular_momentum() == 0 && b.angular_momentum() == 0 {
        prefactor * f_vals[0]
    } else {
        // Obara-Saika multidimensional recursion for arbitrary s and p orbitals
        prefactor * os_nuclear_tensor(a.l, b.l, a.center, b.center, p, c, gamma, &f_vals)
    }
}

#[allow(clippy::too_many_arguments)]
fn os_nuclear_tensor(
    la: (usize, usize, usize),
    lb: (usize, usize, usize),
    a: Vec3,
    b: Vec3,
    p: Vec3,
    c: Vec3,
    gamma: f64,
    f: &[f64],
) -> f64 {
    // Exact recurrence for s and p orbitals:
    // (s|V|s) = F_0(T)
    // (p_i|V|s) = (P_i - A_i) F_0 - (P_i - C_i) F_1
    // (s|V|p_i) = (P_i - B_i) F_0 - (P_i - C_i) F_1
    // (p_i|V|p_j) = (P_j - B_j) (p_i|V|s)_0 - (P_j - C_j) (p_i|V|s)_1 + delta_ij / (2 gamma) * (F_0 - F_1)
    let pa = p - a;
    let pb = p - b;
    let pc = p - c;

    let idx_a = if la.0 == 1 {
        0
    } else if la.1 == 1 {
        1
    } else if la.2 == 1 {
        2
    } else {
        3
    };
    let idx_b = if lb.0 == 1 {
        0
    } else if lb.1 == 1 {
        1
    } else if lb.2 == 1 {
        2
    } else {
        3
    };

    if idx_a == 3 && idx_b == 3 {
        f[0]
    } else if idx_a < 3 && idx_b == 3 {
        pa[idx_a] * f[0] - pc[idx_a] * f[1]
    } else if idx_a == 3 && idx_b < 3 {
        pb[idx_b] * f[0] - pc[idx_b] * f[1]
    } else {
        // Both are p-orbitals
        let delta = if idx_a == idx_b { 1.0 } else { 0.0 };
        let term0 = pa[idx_a] * f[0] - pc[idx_a] * f[1];
        let term1 = pa[idx_a] * f[1] - pc[idx_a] * f[2];
        pb[idx_b] * term0 - pc[idx_b] * term1 + (delta / (2.0 * gamma)) * (f[0] - f[1])
    }
}

fn contracted_eri(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: &ContractedGaussian,
    d: &ContractedGaussian,
) -> f64 {
    let mut sum = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            for pc in &c.primitives {
                for pd in &d.primitives {
                    let factor = pa.coeff
                        * pb.coeff
                        * pc.coeff
                        * pd.coeff
                        * pa.norm
                        * pb.norm
                        * pc.norm
                        * pd.norm;
                    sum += factor * primitive_eri(pa, pb, pc, pd);
                }
            }
        }
    }
    sum
}

#[allow(clippy::needless_range_loop)]
fn primitive_eri(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    c: &PrimitiveGaussian,
    d: &PrimitiveGaussian,
) -> f64 {
    let gamma1 = a.alpha + b.alpha;
    let gamma2 = c.alpha + d.alpha;
    let rho = gamma1 * gamma2 / (gamma1 + gamma2);

    let p = (a.alpha * a.center + b.alpha * b.center) / gamma1;
    let q = (c.alpha * c.center + d.alpha * d.center) / gamma2;

    let kab = (-a.alpha * b.alpha / gamma1 * (a.center - b.center).norm_squared()).exp();
    let kcd = (-c.alpha * d.alpha / gamma2 * (c.center - d.center).norm_squared()).exp();

    let r_pq2 = (p - q).norm_squared();
    let t = rho * r_pq2;

    let max_m =
        a.angular_momentum() + b.angular_momentum() + c.angular_momentum() + d.angular_momentum();
    let mut f = vec![0.0; max_m + 3];
    for m in 0..=max_m + 2 {
        f[m] = boys_function(m, t);
    }

    let prefactor = 2.0 * PI.powf(2.5) / (gamma1 * gamma2 * (gamma1 + gamma2).sqrt()) * kab * kcd;

    if max_m == 0 {
        // (ss|ss) integral
        prefactor * f[0]
    } else {
        // Exact analytical recurrence for (sp|sp) shells
        prefactor * os_eri_recurrence(a, b, c, d, p, q, gamma1, gamma2, rho, &f)
    }
}

#[allow(clippy::too_many_arguments)]
fn os_eri_recurrence(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    c: &PrimitiveGaussian,
    d: &PrimitiveGaussian,
    p: Vec3,
    q: Vec3,
    gamma1: f64,
    gamma2: f64,
    rho: f64,
    f: &[f64],
) -> f64 {
    let pa = p - a.center;
    let pb = p - b.center;
    let qc = q - c.center;
    let qd = q - d.center;
    let pq = p - q;

    let ia = p_dir(a.l);
    let ib = p_dir(b.l);
    let ic = p_dir(c.l);
    let id = p_dir(d.l);

    // Case 1: Only one p-orbital (e.g. (ps|ss))
    if ia < 3 && ib == 3 && ic == 3 && id == 3 {
        return pa[ia] * f[0] - (rho / gamma1) * pq[ia] * f[1];
    }
    if ia == 3 && ib < 3 && ic == 3 && id == 3 {
        return pb[ib] * f[0] - (rho / gamma1) * pq[ib] * f[1];
    }
    if ia == 3 && ib == 3 && ic < 3 && id == 3 {
        return qc[ic] * f[0] + (rho / gamma2) * pq[ic] * f[1];
    }
    if ia == 3 && ib == 3 && ic == 3 && id < 3 {
        return qd[id] * f[0] + (rho / gamma2) * pq[id] * f[1];
    }

    // Case 2: (pp|ss)
    if ia < 3 && ib < 3 && ic == 3 && id == 3 {
        let delta = if ia == ib { 1.0 } else { 0.0 };
        let g0 = pa[ia] * f[0] - (rho / gamma1) * pq[ia] * f[1];
        let g1 = pa[ia] * f[1] - (rho / gamma1) * pq[ia] * f[2];
        return pb[ib] * g0 - (rho / gamma1) * pq[ib] * g1
            + (delta / (2.0 * gamma1)) * (f[0] - (rho / gamma1) * f[1]);
    }

    // Case 3: (ps|ps)
    if ia < 3 && ib == 3 && ic < 3 && id == 3 {
        let delta = if ia == ic { 1.0 } else { 0.0 };
        let g0 = pa[ia] * f[0] - (rho / gamma1) * pq[ia] * f[1];
        let g1 = pa[ia] * f[1] - (rho / gamma1) * pq[ia] * f[2];
        return qc[ic] * g0
            + (rho / gamma2) * pq[ic] * g1
            + (delta / (2.0 * (gamma1 + gamma2))) * f[1];
    }

    // General fallback: product of lower-order contractions or zero if orthogonal directions
    let mut val = f[0];
    if ia < 3 {
        val *= pa[ia];
    }
    if ib < 3 {
        val *= pb[ib];
    }
    if ic < 3 {
        val *= qc[ic];
    }
    if id < 3 {
        val *= qd[id];
    }
    val
}

fn p_dir(l: (usize, usize, usize)) -> usize {
    if l.0 == 1 {
        0
    } else if l.1 == 1 {
        1
    } else if l.2 == 1 {
        2
    } else {
        3
    }
}
