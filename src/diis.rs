//! Direct Inversion in the Iterative Subspace (DIIS / Pulay 1980)
//! Accelerates Self-Consistent Field (SCF) convergence and prevents oscillation.

use crate::types::Mat;

#[derive(Debug, Clone)]
pub struct DIIS {
    pub max_space: usize,
    fock_history: Vec<Mat>,
    error_history: Vec<Mat>,
}

impl DIIS {
    pub fn new(max_space: usize) -> Self {
        Self {
            max_space: max_space.max(2),
            fock_history: Vec::new(),
            error_history: Vec::new(),
        }
    }

    /// Reset DIIS history
    pub fn reset(&mut self) {
        self.fock_history.clear();
        self.error_history.clear();
    }

    /// Computes the commutator error matrix in the orthonormal basis:
    /// e = S^(-1/2) * (F * P * S - S * P * F) * S^(-1/2)
    pub fn compute_error(f: &Mat, p: &Mat, s: &Mat, s_half_inv: &Mat) -> Mat {
        let fps = f * p * s;
        let spf = s * p * f;
        let comm = fps - spf;
        s_half_inv * comm * s_half_inv
    }

    /// Pushes a new (Fock, Error) pair and computes the extrapolated Fock matrix.
    pub fn extrapolate(&mut self, f: &Mat, err: &Mat) -> (Mat, f64) {
        let err_norm = (err.dot(err)).sqrt();

        self.fock_history.push(f.clone());
        self.error_history.push(err.clone());

        if self.fock_history.len() > self.max_space {
            self.fock_history.remove(0);
            self.error_history.remove(0);
        }

        let m = self.fock_history.len();
        if m < 2 {
            return (f.clone(), err_norm);
        }

        // Build Pulay matrix B of dimension (m + 1) x (m + 1)
        // [ B_{ij}  -1 ] [ c_j ] = [ 0 ]
        // [ -1^T     0 ] [  λ  ]   [ -1 ]
        let mut b = Mat::zeros(m + 1, m + 1);
        let mut rhs = nalgebra::DVector::zeros(m + 1);

        for i in 0..m {
            for j in 0..=i {
                let bij = self.error_history[i].dot(&self.error_history[j]);
                b[(i, j)] = bij;
                b[(j, i)] = bij;
            }
            b[(i, m)] = -1.0;
            b[(m, i)] = -1.0;
        }
        b[(m, m)] = 0.0;
        rhs[m] = -1.0;

        // Solve linear system: B * x = rhs
        let decomp = b.lu();
        if let Some(solution) = decomp.solve(&rhs) {
            let mut f_extrap = Mat::zeros(f.nrows(), f.ncols());
            for i in 0..m {
                f_extrap += &self.fock_history[i] * solution[i];
            }
            (f_extrap, err_norm)
        } else {
            // SVD fallback or return unmodified Fock matrix if singular
            (f.clone(), err_norm)
        }
    }
}
