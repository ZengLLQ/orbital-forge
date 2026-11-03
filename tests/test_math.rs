use approx::assert_relative_eq;
use orbital_forge::math::{binomial, boys_f0, boys_function, double_factorial, erf, gaussian_norm};

#[test]
fn test_double_factorial() {
    assert_eq!(double_factorial(-1), 1.0);
    assert_eq!(double_factorial(0), 1.0);
    assert_eq!(double_factorial(1), 1.0);
    assert_eq!(double_factorial(2), 2.0);
    assert_eq!(double_factorial(3), 3.0);
    assert_eq!(double_factorial(4), 8.0);
    assert_eq!(double_factorial(5), 15.0);
    assert_eq!(double_factorial(6), 48.0);
    assert_eq!(double_factorial(7), 105.0);
}

#[test]
fn test_binomial() {
    assert_eq!(binomial(0, 0), 1.0);
    assert_eq!(binomial(4, 2), 6.0);
    assert_eq!(binomial(5, 3), 10.0);
    assert_eq!(binomial(6, 0), 1.0);
    assert_eq!(binomial(6, 6), 1.0);
}

#[test]
fn test_erf_properties() {
    assert_relative_eq!(erf(0.0), 0.0, epsilon = 1e-12);
    assert_relative_eq!(erf(1.0), 0.84270079, epsilon = 1e-5);
    assert_relative_eq!(erf(-1.0), -0.84270079, epsilon = 1e-5);
    assert_relative_eq!(erf(5.0), 1.0, epsilon = 1e-6);
}

#[test]
fn test_boys_function_properties() {
    // F_n(0) = 1 / (2n + 1)
    for n in 0..5 {
        let expected = 1.0 / (2.0 * n as f64 + 1.0);
        assert_relative_eq!(boys_function(n, 0.0), expected, epsilon = 1e-8);
    }

    // F_0(T) comparison with erf formulation
    for &t in &[0.05, 0.5, 1.5, 5.0, 15.0] {
        let f0_quad = boys_function(0, t);
        let f0_direct = boys_f0(t);
        assert_relative_eq!(f0_quad, f0_direct, epsilon = 1e-6);
    }

    // Monotonicity: F_n(T1) > F_n(T2) for T1 < T2
    let val1 = boys_function(1, 1.0);
    let val2 = boys_function(1, 2.0);
    assert!(val1 > val2);
}

#[test]
fn test_gaussian_norm() {
    let alpha = 1.25;
    // s-orbital (0, 0, 0)
    let norm_s = gaussian_norm(alpha, 0, 0, 0);
    assert!(norm_s > 0.0);

    // p-orbital (1, 0, 0)
    let norm_p = gaussian_norm(alpha, 1, 0, 0);
    assert!(norm_p > 0.0);
}
