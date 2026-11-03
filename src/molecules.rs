//! Canonical molecular geometries and XYZ coordinate parsing.

use crate::types::{Atom, Molecule, ANGSTROM_TO_BOHR};

/// Hydrogen molecule (H2) along the z-axis.
/// Default bond distance: 1.401 Bohr (0.7414 Ångström).
pub fn h2_molecule(bond_length_bohr: Option<f64>) -> Molecule {
    let r = bond_length_bohr.unwrap_or(1.40107);
    let half_r = r * 0.5;
    let atoms = vec![
        Atom::new("H", 1, 0.0, 0.0, -half_r),
        Atom::new("H", 1, 0.0, 0.0, half_r),
    ];
    Molecule::new("Hydrogen (H2)", atoms, 0, 1)
}

/// Helium hydride cation (HeH+).
/// Default bond distance: 1.4632 Bohr (0.7743 Ångström).
pub fn heh_plus(bond_length_bohr: Option<f64>) -> Molecule {
    let r = bond_length_bohr.unwrap_or(1.4632);
    let atoms = vec![
        Atom::new("He", 2, 0.0, 0.0, 0.0),
        Atom::new("H", 1, 0.0, 0.0, r),
    ];
    Molecule::new("Helium Hydride (HeH+)", atoms, 1, 1)
}

/// Water molecule (H2O) at equilibrium geometry:
/// R(OH) = 0.957 Å, Angle(HOH) = 104.5 degrees.
pub fn water_molecule() -> Molecule {
    let r_oh_ang = 0.9572;
    let angle_deg = 104.52;
    let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
    let half_angle = angle_rad * 0.5;

    let r_bohr = r_oh_ang * ANGSTROM_TO_BOHR;
    let h_y = r_bohr * half_angle.sin();
    let h_z = r_bohr * half_angle.cos();

    let atoms = vec![
        Atom::new("O", 8, 0.0, 0.0, 0.0),
        Atom::new("H", 1, 0.0, h_y, h_z),
        Atom::new("H", 1, 0.0, -h_y, h_z),
    ];
    Molecule::new("Water (H2O)", atoms, 0, 1)
}

/// Linear trihydrogen cation (H3+).
pub fn linear_h3_plus() -> Molecule {
    let r = 1.6; // Bohr
    let atoms = vec![
        Atom::new("H", 1, 0.0, 0.0, -r),
        Atom::new("H", 1, 0.0, 0.0, 0.0),
        Atom::new("H", 1, 0.0, 0.0, r),
    ];
    Molecule::new("Trihydrogen Cation (H3+)", atoms, 1, 1)
}

/// Parses an XYZ format molecular geometry string (coordinates in Ångströms).
pub fn parse_xyz(
    name: &str,
    content: &str,
    charge: i32,
    multiplicity: usize,
) -> Result<Molecule, String> {
    let lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return Err("Empty XYZ content".to_string());
    }

    let n_atoms: usize = lines[0]
        .parse()
        .map_err(|e| format!("Invalid atom count: {}", e))?;
    let mut atoms = Vec::with_capacity(n_atoms);

    // Line 1 is comment/description, data starts at line 2
    let start_idx = if lines.len() > n_atoms + 1 { 2 } else { 1 };

    for line in lines.iter().skip(start_idx).take(n_atoms) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("Invalid atom line format: {}", line));
        }

        let symbol = parts[0];
        let x: f64 = parts[1].parse().map_err(|e| format!("Invalid X: {}", e))?;
        let y: f64 = parts[2].parse().map_err(|e| format!("Invalid Y: {}", e))?;
        let z: f64 = parts[3].parse().map_err(|e| format!("Invalid Z: {}", e))?;

        let z_num = match symbol.to_uppercase().as_str() {
            "H" => 1,
            "HE" => 2,
            "C" => 6,
            "N" => 7,
            "O" => 8,
            "F" => 9,
            _ => return Err(format!("Unsupported element: {}", symbol)),
        };

        atoms.push(Atom::from_angstrom(symbol, z_num, x, y, z));
    }

    Ok(Molecule::new(name, atoms, charge, multiplicity))
}
