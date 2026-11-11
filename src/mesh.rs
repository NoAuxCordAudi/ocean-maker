//! Grid meshes built from a [`DisplacementField`].
//!
//! The surface lies in the XY plane with Z as height.

use anyhow::{ensure, Result};
use glam::{Vec2, Vec3};

use crate::tessendorf::DisplacementField;

/// Largest allowed |difference| between wrapped edge vertices (should be ~f32 rounding).
pub const SEAM_TOLERANCE: f32 = 1e-4;

/// Dimensions and tessellation of a rectangular grid centred on the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSpec {
    width: f32,
    length: f32,
    width_segments: usize,
    length_segments: usize,
}

impl GridSpec {
    /// # Errors
    /// If a size is not finite and positive, or a segment count is zero.
    pub fn new(width: f32, length: f32, width_segments: usize, length_segments: usize) -> Result<Self> {
        ensure!(width.is_finite() && width > 0.0, "width must be > 0, got {width}");
        ensure!(length.is_finite() && length > 0.0, "length must be > 0, got {length}");
        ensure!(width_segments >= 1, "width segments must be >= 1");
        ensure!(length_segments >= 1, "length segments must be >= 1");
        Ok(Self { width, length, width_segments, length_segments })
    }

    pub fn width(&self) -> f32 {
        self.width
    }
    pub fn length(&self) -> f32 {
        self.length
    }
    pub fn width_segments(&self) -> usize {
        self.width_segments
    }
    pub fn length_segments(&self) -> usize {
        self.length_segments
    }

    fn stride(&self) -> usize {
        self.width_segments + 1
    }

    fn vertex_count(&self) -> usize {
        self.stride() * (self.length_segments + 1)
    }

    /// Normalised grid coordinates (u, v) in [0, 1] of vertex `(ix, iy)`.
    fn uv(&self, ix: usize, iy: usize) -> (f64, f64) {
        (ix as f64 / self.width_segments as f64, iy as f64 / self.length_segments as f64)
    }

    /// Undisplaced XY position of a vertex from its (u, v).
    fn xy(&self, u: f64, v: f64) -> (f32, f32) {
        (u as f32 * self.width - self.width / 2.0, v as f32 * self.length - self.length / 2.0)
    }

    fn indices(&self) -> Vec<u32> {
        let stride = self.stride();
        let mut indices = Vec::with_capacity(self.width_segments * self.length_segments * 6);
        for iy in 0..self.length_segments {
            for ix in 0..self.width_segments {
                let a = (iy * stride + ix) as u32;
                let b = a + 1;
                let d = ((iy + 1) * stride + ix) as u32;
                let c = d + 1;
                indices.extend_from_slice(&[a, b, d, b, c, d]);
            }
        }
        indices
    }
}

/// Vertex buffers of a triangle mesh. All per-vertex vectors have equal length.
#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// (min, max) of the vertex Z coordinates.
    pub fn height_range(&self) -> (f32, f32) {
        self.positions
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.z), hi.max(p.z)))
    }
}

/// A flat grid at Z = 0 with upward normals: the rest pose for morph targets.
pub fn flat_grid(grid: &GridSpec) -> MeshData {
    let mut positions = Vec::with_capacity(grid.vertex_count());
    let mut uvs = Vec::with_capacity(grid.vertex_count());
    for iy in 0..=grid.length_segments {
        for ix in 0..=grid.width_segments {
            let (u, v) = grid.uv(ix, iy);
            let (x, y) = grid.xy(u, v);
            positions.push(Vec3::new(x, y, 0.0));
            uvs.push(Vec2::new(u as f32, v as f32));
        }
    }
    MeshData {
        normals: vec![Vec3::Z; positions.len()],
        positions,
        uvs,
        indices: grid.indices(),
    }
}

/// Build the ocean surface from a displacement field.
///
/// The field is periodic with period `field_size` (world units). Each vertex samples the
/// field at its physical position with wrap-around interpolation, so when
/// `width == length == field_size` the last row/column of vertices samples exactly the same
/// field point as the first. The mesh is then seamlessly tileable with no edge patching.
pub fn ocean_mesh(grid: &GridSpec, field_size: f32, field: &DisplacementField) -> MeshData {
    let n = field.resolution as f64;
    // Grid-cell span of the whole mesh along each axis (== n when the mesh covers one period).
    let span_x = n * (grid.width as f64 / field_size as f64);
    let span_y = n * (grid.length as f64 / field_size as f64);

    let mut mesh = flat_grid(grid);
    for iy in 0..=grid.length_segments {
        for ix in 0..=grid.width_segments {
            let (u, v) = grid.uv(ix, iy);
            let s = field.sample_periodic(u * span_x, v * span_y);
            let idx = iy * grid.stride() + ix;
            mesh.positions[idx] += Vec3::new(s.displacement.x, s.displacement.y, s.height);
            mesh.normals[idx] = Vec3::from(s.normal);
        }
    }
    mesh
}

/// Largest deviation between the mesh's opposite edges, comparing each edge vertex with its
/// counterpart shifted by one tile. Covers x, y, z and the normal, so choppy displacement
/// is checked too.
pub fn max_seam_error(mesh: &MeshData, grid: &GridSpec) -> f32 {
    let stride = grid.stride();
    let mut max_err: f32 = 0.0;
    let mut compare = |a: usize, b: usize, shift: Vec3| {
        let dp = mesh.positions[b] - shift - mesh.positions[a];
        let dn = mesh.normals[b] - mesh.normals[a];
        max_err = max_err.max(dp.abs().max_element()).max(dn.abs().max_element());
    };
    for iy in 0..=grid.length_segments {
        compare(iy * stride, iy * stride + grid.width_segments, Vec3::new(grid.width, 0.0, 0.0));
    }
    for ix in 0..=grid.width_segments {
        compare(ix, grid.length_segments * stride + ix, Vec3::new(0.0, grid.length, 0.0));
    }
    max_err
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tessendorf::{compute_wave_field, OceanParams};

    fn build(w: f32, l: f32, ws: usize, ls: usize) -> (MeshData, GridSpec) {
        let size = w.max(l);
        let params = OceanParams { resolution: ws.max(ls), size, choppiness: 0.8, ..Default::default() };
        let field = compute_wave_field(&params, 0.7);
        let grid = GridSpec::new(w, l, ws, ls).unwrap();
        (ocean_mesh(&grid, size, &field), grid)
    }

    #[test]
    fn square_mesh_is_seamless() {
        let (m, g) = build(20.0, 20.0, 64, 64);
        assert!(max_seam_error(&m, &g) <= SEAM_TOLERANCE);
    }

    #[test]
    fn seamless_when_segments_differ_from_field_resolution() {
        let (m, g) = build(20.0, 20.0, 48, 32);
        assert!(max_seam_error(&m, &g) <= SEAM_TOLERANCE);
    }

    #[test]
    fn seam_check_detects_a_broken_edge() {
        let (mut m, g) = build(20.0, 20.0, 16, 16);
        m.positions[16].z += 0.5;
        assert!(max_seam_error(&m, &g) > SEAM_TOLERANCE);
    }

    #[test]
    fn edge_vertices_are_not_flattened() {
        let (m, _) = build(20.0, 20.0, 64, 64);
        assert!(m.normals[..65].iter().any(|n| n.x.abs() > 1e-3 || n.y.abs() > 1e-3));
    }

    #[test]
    fn flat_grid_matches_ocean_topology_and_xy_without_choppiness() {
        let grid = GridSpec::new(20.0, 20.0, 16, 16).unwrap();
        let params = OceanParams { resolution: 16, size: 20.0, choppiness: 0.0, ..Default::default() };
        let m = ocean_mesh(&grid, 20.0, &compute_wave_field(&params, 0.0));
        let flat = flat_grid(&grid);
        assert_eq!(m.indices, flat.indices);
        assert_eq!(m.uvs, flat.uvs);
        for (a, b) in m.positions.iter().zip(&flat.positions) {
            assert_eq!((a.x, a.y), (b.x, b.y));
        }
    }

    #[test]
    fn grid_rejects_degenerate_input() {
        assert!(GridSpec::new(0.0, 1.0, 1, 1).is_err());
        assert!(GridSpec::new(1.0, f32::NAN, 1, 1).is_err());
        assert!(GridSpec::new(1.0, 1.0, 0, 1).is_err());
        assert!(GridSpec::new(1.0, 1.0, 1, 0).is_err());
    }
}
