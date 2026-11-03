//! Standard Pople STO-3G Gaussian Basis Sets for First and Second Row Elements.

use crate::math::gaussian_norm;
use crate::types::{BasisSet, ContractedGaussian, Molecule, PrimitiveGaussian, Vec3};

/// Constructs an STO-3G basis set for the given molecule.
pub fn build_sto3g_basis(molecule: &Molecule) -> BasisSet {
    let mut functions = Vec::new();

    for (atom_idx, atom) in molecule.atoms.iter().enumerate() {
        match atom.atomic_number {
            1 => {
                // Hydrogen: 1s orbital
                functions.push(create_contracted_orbital(
                    atom_idx,
                    atom.position,
                    (0, 0, 0),
                    &[3.425_250_91, 0.623_913_73, 0.168_855_40],
                    &[0.154_328_97, 0.535_328_14, 0.444_634_54],
                ));
            }
            2 => {
                // Helium: 1s orbital
                functions.push(create_contracted_orbital(
                    atom_idx,
                    atom.position,
                    (0, 0, 0),
                    &[6.362_421_39, 1.158_923_00, 0.313_649_79],
                    &[0.154_328_97, 0.535_328_14, 0.444_634_54],
                ));
            }
            6 => {
                // Carbon: 1s, 2s, 2px, 2py, 2pz
                add_row2_sto3g(
                    &mut functions,
                    atom_idx,
                    atom.position,
                    &[71.616_837_0, 13.045_096_0, 3.530_512_2],
                    &[0.154_328_97, 0.535_328_14, 0.444_634_54],
                    &[2.941_249_4, 0.683_483_1, 0.222_289_9],
                    &[-0.099_967_23, 0.399_512_83, 0.700_115_47],
                    &[0.155_916_27, 0.607_683_72, 0.391_957_39],
                );
            }
            7 => {
                // Nitrogen: 1s, 2s, 2px, 2py, 2pz
                add_row2_sto3g(
                    &mut functions,
                    atom_idx,
                    atom.position,
                    &[99.106_169, 18.052_312, 4.885_660_2],
                    &[0.154_328_97, 0.535_328_14, 0.444_634_54],
                    &[3.780_455_9, 0.878_496_6, 0.285_714_4],
                    &[-0.099_967_23, 0.399_512_83, 0.700_115_47],
                    &[0.155_916_27, 0.607_683_72, 0.391_957_39],
                );
            }
            8 => {
                // Oxygen: 1s, 2s, 2px, 2py, 2pz
                add_row2_sto3g(
                    &mut functions,
                    atom_idx,
                    atom.position,
                    &[130.709_32, 23.808_861, 6.443_608_3],
                    &[0.154_328_97, 0.535_328_14, 0.444_634_54],
                    &[5.033_151_3, 1.169_596_1, 0.380_389_0],
                    &[-0.099_967_23, 0.399_512_83, 0.700_115_47],
                    &[0.155_916_27, 0.607_683_72, 0.391_957_39],
                );
            }
            9 => {
                // Fluorine: 1s, 2s, 2px, 2py, 2pz
                add_row2_sto3g(
                    &mut functions,
                    atom_idx,
                    atom.position,
                    &[166.679_14, 30.360_811, 8.216_820_8],
                    &[0.154_328_97, 0.535_328_14, 0.444_634_54],
                    &[6.464_803_2, 1.502_281_2, 0.488_588_5],
                    &[-0.099_967_23, 0.399_512_83, 0.700_115_47],
                    &[0.155_916_27, 0.607_683_72, 0.391_957_39],
                );
            }
            _ => panic!(
                "STO-3G parameters not implemented for element Z={}",
                atom.atomic_number
            ),
        }
    }

    BasisSet {
        name: "STO-3G".to_string(),
        functions,
    }
}

#[allow(clippy::too_many_arguments)]
fn add_row2_sto3g(
    funcs: &mut Vec<ContractedGaussian>,
    atom_idx: usize,
    pos: Vec3,
    alphas_1s: &[f64; 3],
    coeffs_1s: &[f64; 3],
    alphas_2sp: &[f64; 3],
    coeffs_2s: &[f64; 3],
    coeffs_2p: &[f64; 3],
) {
    // 1s
    funcs.push(create_contracted_orbital(
        atom_idx,
        pos,
        (0, 0, 0),
        alphas_1s,
        coeffs_1s,
    ));
    // 2s
    funcs.push(create_contracted_orbital(
        atom_idx,
        pos,
        (0, 0, 0),
        alphas_2sp,
        coeffs_2s,
    ));
    // 2px (1, 0, 0)
    funcs.push(create_contracted_orbital(
        atom_idx,
        pos,
        (1, 0, 0),
        alphas_2sp,
        coeffs_2p,
    ));
    // 2py (0, 1, 0)
    funcs.push(create_contracted_orbital(
        atom_idx,
        pos,
        (0, 1, 0),
        alphas_2sp,
        coeffs_2p,
    ));
    // 2pz (0, 0, 1)
    funcs.push(create_contracted_orbital(
        atom_idx,
        pos,
        (0, 0, 1),
        alphas_2sp,
        coeffs_2p,
    ));
}

fn create_contracted_orbital(
    atom_index: usize,
    center: Vec3,
    l: (usize, usize, usize),
    alphas: &[f64],
    coeffs: &[f64],
) -> ContractedGaussian {
    let mut primitives = Vec::new();
    let n_prim = alphas.len();

    // 1. Primitive Gaussians with uncontracted norm
    for (&alpha, &coeff) in alphas.iter().zip(coeffs.iter()) {
        let norm = gaussian_norm(alpha, l.0, l.1, l.2);
        primitives.push(PrimitiveGaussian {
            alpha,
            coeff,
            norm,
            center,
            l,
        });
    }

    // 2. Compute self-overlap for contraction normalization factor N_c
    let mut s_tot = 0.0;
    for i in 0..n_prim {
        for j in 0..n_prim {
            let a_i = primitives[i].alpha;
            let a_j = primitives[j].alpha;
            let gamma = a_i + a_j;

            // Analytic self-overlap between concentric Cartesian primitives
            let s_prim = primitives[i].norm
                * primitives[j].norm
                * primitive_concentric_overlap(l.0, a_i, a_j)
                * primitive_concentric_overlap(l.1, a_i, a_j)
                * primitive_concentric_overlap(l.2, a_i, a_j)
                * (std::f64::consts::PI / gamma).powf(1.5);

            s_tot += primitives[i].coeff * primitives[j].coeff * s_prim;
        }
    }

    let n_c = 1.0 / s_tot.sqrt();
    for prim in &mut primitives {
        prim.coeff *= n_c;
    }

    ContractedGaussian {
        atom_index,
        center,
        l,
        primitives,
    }
}

fn primitive_concentric_overlap(l: usize, a: f64, b: f64) -> f64 {
    let gamma = a + b;
    if l == 0 {
        1.0
    } else if l == 1 {
        0.5 / gamma
    } else if l == 2 {
        0.75 / (gamma * gamma)
    } else {
        crate::math::double_factorial(2 * l as i64 - 1) / (2.0 * gamma).powi(l as i32)
    }
}
