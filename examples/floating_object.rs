//! Places a box-shaped "boat" on the ocean surface, tilted to the local normal.
//! Writes `ocean.obj` and `boat.obj` to `./floating_output`.

mod common;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use glam::{Quat, Vec3};
use ocean_generator::tessendorf::{compute_wave_field, OceanParams};

/// Axis-aligned box centred at the origin, with outward-facing triangles.
fn box_mesh(half: Vec3) -> common::Mesh {
    let mut positions = Vec::new();
    for &z in &[-half.z, half.z] {
        for &(x, y) in &[(-half.x, -half.y), (half.x, -half.y), (half.x, half.y), (-half.x, half.y)] {
            positions.push(Vec3::new(x, y, z));
        }
    }
    let triangles = vec![
        [0, 2, 1], [0, 3, 2], // bottom
        [4, 5, 6], [4, 6, 7], // top
        [0, 1, 5], [0, 5, 4], // -y
        [1, 2, 6], [1, 6, 5], // +x
        [2, 3, 7], [2, 7, 6], // +y
        [3, 0, 4], [3, 4, 7], // -x
    ];
    let normals = positions.iter().map(|p| p.normalize()).collect();
    common::Mesh { positions, normals, triangles }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 30.0;
    let params = OceanParams {
        resolution: 64,
        size,
        wind_speed: 6.0,
        amplitude: 0.5,
        wind_direction: glam::Vec2::new(30f32.to_radians().cos(), 30f32.to_radians().sin()),
        seed: 123,
        ..OceanParams::default()
    };
    let field = compute_wave_field(&params, 0.0);
    let ocean = common::ocean_mesh(&field, size);

    // Sit the boat on the surface at (5, 2), rotated so its up axis follows the surface normal.
    let (bx, by) = (5.0, 2.0);
    let (surface_z, normal) = common::height_at(&field, size, bx, by);
    let rotation = Quat::from_rotation_arc(Vec3::Z, normal);
    let half = Vec3::new(1.5, 0.75, 0.4);
    let mut boat = box_mesh(half);
    let center = Vec3::new(bx, by, surface_z) + normal * half.z;
    for p in &mut boat.positions {
        *p = rotation * *p + center;
    }
    for n in &mut boat.normals {
        *n = rotation * *n;
    }

    let output_dir = PathBuf::from("./floating_output");
    std::fs::create_dir_all(&output_dir)?;
    common::write_obj(&ocean, &mut BufWriter::new(File::create(output_dir.join("ocean.obj"))?))?;
    common::write_obj(&boat, &mut BufWriter::new(File::create(output_dir.join("boat.obj"))?))?;
    println!("Wrote ocean.obj and boat.obj to {}", output_dir.display());
    Ok(())
}
