//! Helpers shared by the examples: build a mesh from a displacement field and write OBJ.
#![allow(dead_code)] // each example uses a different subset of these helpers


use std::io::{self, Write};

use glam::Vec3;
use ocean_generator::tessendorf::DisplacementField;

/// Triangle mesh in the XY plane with Z up.
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// Triangles as 0-based vertex indices.
    pub triangles: Vec<[u32; 3]>,
}

/// Build one full period of the ocean (`resolution + 1` vertices per side; the
/// last row/column duplicates the first so tiles join seamlessly), centred on the origin.
pub fn ocean_mesh(field: &DisplacementField, size: f32) -> Mesh {
    let n = field.resolution;
    let stride = n + 1;
    let mut positions = Vec::with_capacity(stride * stride);
    let mut normals = Vec::with_capacity(stride * stride);
    for iy in 0..=n {
        for ix in 0..=n {
            let s = field.sample_periodic(ix as f64, iy as f64);
            let x = ix as f32 / n as f32 * size - size / 2.0;
            let y = iy as f32 / n as f32 * size - size / 2.0;
            positions.push(Vec3::new(x + s.displacement.x, y + s.displacement.y, s.height));
            normals.push(Vec3::from(s.normal));
        }
    }
    let mut triangles = Vec::with_capacity(n * n * 2);
    for iy in 0..n {
        for ix in 0..n {
            let a = (iy * stride + ix) as u32;
            let b = a + 1;
            let d = a + stride as u32;
            let c = d + 1;
            triangles.push([a, b, d]);
            triangles.push([b, c, d]);
        }
    }
    Mesh { positions, normals, triangles }
}

/// Surface height at world position `(x, y)` (ignores horizontal choppy displacement).
pub fn height_at(field: &DisplacementField, size: f32, x: f32, y: f32) -> (f32, Vec3) {
    let n = field.resolution as f64;
    let u = (x as f64 / size as f64 + 0.5) * n;
    let v = (y as f64 / size as f64 + 0.5) * n;
    let s = field.sample_periodic(u, v);
    (s.height, Vec3::from(s.normal))
}

/// Write a mesh as Wavefront OBJ.
pub fn write_obj(mesh: &Mesh, out: &mut impl Write) -> io::Result<()> {
    for p in &mesh.positions {
        writeln!(out, "v {} {} {}", p.x, p.y, p.z)?;
    }
    for n in &mesh.normals {
        writeln!(out, "vn {} {} {}", n.x, n.y, n.z)?;
    }
    for [a, b, c] in &mesh.triangles {
        let (a, b, c) = (a + 1, b + 1, c + 1);
        writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}")?;
    }
    Ok(())
}
