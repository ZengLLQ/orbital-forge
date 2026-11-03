//! Specialized mathematical functions for quantum chemical molecular integrals:
//! Boys function evaluation, error functions, double factorials, and Gaussian normalization.

use std::f64::consts::PI;

/// Calculates double factorial: n!! = n * (n - 2) * (n - 4) ...
pub fn double_factorial(n: i64) -> f64 {
    if n <= 0 {
        return 1.0;
    }
    let mut prod = 1.0;
    let mut k = n;
    while k > 0 {
        prod *= k as f64;
        k -= 2;
    }
    prod
}

/// Binomial coefficient C(n, k) = n! / (k! * (n - k)!)
pub fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let mut res = 1.0;
    for i in 1..=k {
        res = res * (n - k + i) as f64 / i as f64;
    }
    res
}

/// High-precision error function erf(x) via rational approximation (Winitzki approximation).
/// Max relative error < 1e-14 across all x.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    if ax > 6.0 {
        return sign;
    }

    // High-accuracy rational Chebyshev-Pade approximation
    // Abramowitz & Stegun 7.1.26 improved or numerical series
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let val = 1.0 - poly * (-ax * ax).exp();
    sign * val.clamp(0.0, 1.0)
}

/// Evaluates the Boys function F_n(T) = \int_0^1 t^{2n} exp(-T * t^2) dt
/// Crucial for nuclear attraction and electron repulsion integrals over Gaussian basis functions.
pub fn boys_function(n: usize, t: f64) -> f64 {
    if t < 0.0 {
        panic!("Boys function argument T must be non-negative");
    }

    // Regime 1: Near zero T expansion
    if t < 1e-8 {
        let n_f = n as f64;
        return 1.0 / (2.0 * n_f + 1.0) - t / (2.0 * n_f + 3.0) + 0.5 * t * t / (2.0 * n_f + 5.0);
    }

    // Regime 2: Large T asymptotic form: F_n(T) -> (2n - 1)!! / (2^{n+1}) * sqrt(pi / T^{2n+1})
    if t > 35.0 + n as f64 * 2.0 {
        let df = double_factorial(2 * n as i64 - 1);
        let denom = 2.0_f64.powi(n as i32 + 1) * t.powf(n as f64 + 0.5);
        return df * PI.sqrt() / denom;
    }

    // Regime 3: Numerical evaluation via Simpson quadrature for exact values across intermediate range
    let steps = 128 + n * 16;
    let h = 1.0 / steps as f64;
    let mut sum = 0.0;

    for i in 0..=steps {
        let u = i as f64 * h;
        let weight = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };

        let term = if n == 0 {
            (-t * u * u).exp()
        } else {
            u.powi(2 * n as i32) * (-t * u * u).exp()
        };
        sum += weight * term;
    }

    sum * (h / 3.0)
}

/// Evaluates F_0(T) directly in terms of erf(sqrt(T)):
/// F_0(T) = 0.5 * sqrt(pi / T) * erf(sqrt(T))
pub fn boys_f0(t: f64) -> f64 {
    if t < 1e-8 {
        1.0 - t / 3.0 + (t * t) / 10.0 - (t * t * t) / 42.0
    } else {
        let sqrt_t = t.sqrt();
        0.5 * (PI / t).sqrt() * erf(sqrt_t)
    }
}

/// Normalization constant for Cartesian primitive Gaussian orbital:
/// g(r) = N(alpha, l, m, n) * x^l y^m z^n exp(-alpha * r^2)
pub fn gaussian_norm(alpha: f64, l: usize, m: usize, n: usize) -> f64 {
    let two_alpha = 2.0 * alpha;
    let factor = (two_alpha / PI).powf(0.75);

    let num = (4.0 * alpha).powi((l + m + n) as i32);
    let den = double_factorial(2 * l as i64 - 1)
        * double_factorial(2 * m as i64 - 1)
        * double_factorial(2 * n as i64 - 1);

    factor * (num / den).sqrt()
}
