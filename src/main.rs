use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod tessendorf;

// Use mesh-tools compatibility types
use mesh_tools::compat::{Point3, Vector2, Vector3};

/// Command-line tool to generate an ocean surface using Tessendorf algorithm
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Width of the ocean surface
    #[arg(short, long, default_value_t = 20.0)]
    width: f32,

    /// Length of the ocean surface
    #[arg(short, long, default_value_t = 20.0)]
    length: f32,

    /// Number of segments along the width
    #[arg(long, default_value_t = 128)]
    width_segments: usize,

    /// Number of segments along the length
    #[arg(long, default_value_t = 128)]
    length_segments: usize,

    /// Wind speed for wave generation
    #[arg(long, default_value_t = 5.0)]
    wind_speed: f32,

    /// Wind direction X component (ignored if --wind-angle is set)
    #[arg(long, default_value_t = 1.0, allow_negative_numbers = true)]
    wind_dir_x: f32,

    /// Wind direction Y component (ignored if --wind-angle is set)
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    wind_dir_y: f32,

    /// Wind direction as angle in degrees (0 = +X, 90 = +Y).
    /// When provided, overrides --wind-dir-x / --wind-dir-y.
    #[arg(long, allow_negative_numbers = true)]
    wind_angle: Option<f32>,

    /// Wave amplitude
    #[arg(short, long, default_value_t = 1.0)]
    amplitude: f32,

    /// Wave choppiness (0.0-1.0)
    #[arg(short, long, default_value_t = 0.5)]
    choppiness: f32,

    /// Random seed for wave generation
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Time parameter for wave animation
    #[arg(short, long, default_value_t = 0.0)]
    time: f32,

    /// Output file path
    #[arg(short, long, default_value = "ocean_surface.glb")]
    output: PathBuf,

    /// Number of tiles in each direction (creates NxN grid)
    #[arg(short = 'g', long, default_value_t = 1)]
    tiles: usize,
}

/// Max allowed |difference| between wrapped edge vertices (should be ~f32 rounding).
const SEAM_TOLERANCE: f32 = 1e-4;

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    println!("Generating Tessendorf ocean surface...");
    println!("Ocean dimensions: {}x{} with {}x{} segments", 
             args.width, args.length, args.width_segments, args.length_segments);
    println!("Wave parameters: wind speed={}, amplitude={}, choppiness={}, time={}", 
             args.wind_speed, args.amplitude, args.choppiness, args.time);

    // Resolve wind direction: --wind-angle wins if provided, else x/y components.
    let wind_direction = match args.wind_angle {
        Some(deg) => {
            let rad = deg.to_radians();
            glam::Vec2::new(rad.cos(), rad.sin())
        }
        None => glam::Vec2::new(args.wind_dir_x, args.wind_dir_y).normalize(),
    };
    println!("Wind direction: ({:.3}, {:.3})", wind_direction.x, wind_direction.y);

    // Create ocean parameters
    let params = tessendorf::OceanParams {
        resolution: args.width_segments.max(args.length_segments),
        size: args.width.max(args.length),
        wind_speed: args.wind_speed,
        wind_direction,
        amplitude: args.amplitude,
        choppiness: args.choppiness,
        gravity: 9.81,
        depth: 200.0,
        seed: args.seed,
    };

    // Generate displacement field using Tessendorf algorithm
    println!("Computing wave field...");
    let displacement = tessendorf::compute_wave_field(&params, args.time);

    // Create a mesh from the displacement field
    println!("Creating mesh...");
    let mesh = create_mesh(
        args.width,
        args.length,
        args.width_segments,
        args.length_segments,
        params.size,
        &displacement,
    );
    let (min_height, max_height) = (mesh.min_height, mesh.max_height);

    // The mesh is one exact period of the field only when it spans the full field size.
    let periodic = args.width == params.size && args.length == params.size;
    if periodic {
        let seam_error = max_seam_error(
            &mesh,
            args.width,
            args.length,
            args.width_segments,
            args.length_segments,
        );
        println!("Seam error across opposite edges: {:.2e}", seam_error);
        anyhow::ensure!(
            seam_error <= SEAM_TOLERANCE,
            "mesh is not seamlessly tileable: seam error {seam_error:.2e} exceeds {SEAM_TOLERANCE:.2e}"
        );
    } else if args.tiles > 1 {
        eprintln!(
            "warning: width ({}) != length ({}), so the mesh is only a window of a {size}x{size} \
             periodic field and will show seams when tiled. Use equal width and length for seamless tiling.",
            args.width, args.length, size = params.size
        );
    }

    // Log min/max z-axis displacement and compare with commanded amplitude
    println!("Z-axis displacement range: {:.6} to {:.6} units", min_height, max_height);
    let displacement_range = max_height - min_height;
    let expected_range = args.amplitude * 2.0; // Peak-to-peak should be ~2x amplitude
    println!("Total displacement range: {:.6} units", displacement_range);
    println!("Expected range (2 * amplitude): {:.6} units", expected_range);
    
    // Check if the actual range is close to the expected range
    let range_ratio = displacement_range / expected_range;
    println!("Ratio of actual/expected range: {:.4}", range_ratio);
    if range_ratio >= 0.75 && range_ratio <= 1.25 {
        println!("✓ Displacement range is reasonably close to commanded amplitude");
    } else {
        println!("! Displacement range differs significantly from commanded amplitude");
    }

    // Export the mesh
    println!("Exporting to {}...", args.output.display());
    println!("Creating {}x{} tiled pattern...", args.tiles, args.tiles);
    export_mesh(&mesh, args.width, args.length, &args.output, args.tiles)?;

    println!("Done!");
    Ok(())
}

/// Mesh buffers produced by [`create_mesh`].
struct MeshData {
    positions: Vec<Point3<f32>>,
    indices: Vec<u32>,
    normals: Vec<Vector3<f32>>,
    uvs: Vec<Vector2<f32>>,
    min_height: f32,
    max_height: f32,
}

/// Create a mesh from the displacement field.
/// Note: Ocean surface is defined in the xy-plane, with z-axis as height.
///
/// The FFT field is periodic with period `field_size` (world units). Each
/// vertex samples the field at its physical position with wrap-around
/// interpolation, so when `width == length == field_size` the last row/column
/// of vertices samples exactly the same field point as the first. The mesh is
/// then seamlessly tileable with no post-hoc edge patching.
fn create_mesh(
    width: f32,
    length: f32,
    width_segments: usize,
    length_segments: usize,
    field_size: f32,
    displacement: &tessendorf::DisplacementField,
) -> MeshData {
    let n = displacement.resolution as f64;
    // Grid-cell span of the whole mesh along each axis (== n when mesh covers one full period).
    let span_x = n * (width as f64 / field_size as f64);
    let span_y = n * (length as f64 / field_size as f64);

    let vertex_count = (width_segments + 1) * (length_segments + 1);
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);
    let mut indices = Vec::with_capacity(width_segments * length_segments * 6);

    let mut min_height = f32::MAX;
    let mut max_height = f32::MIN;

    for iy in 0..=length_segments {
        let v = iy as f64 / length_segments as f64;
        let y = v as f32 * length - length / 2.0;
        for ix in 0..=width_segments {
            let u = ix as f64 / width_segments as f64;
            let x = u as f32 * width - width / 2.0;

            let s = displacement.sample_periodic(u * span_x, v * span_y);

            positions.push(mesh_tools::compat::point3::new(
                x + s.displacement.x,
                y + s.displacement.y,
                s.height,
            ));
            normals.push(mesh_tools::compat::vector3::new(s.normal[0], s.normal[1], s.normal[2]));
            uvs.push(mesh_tools::compat::vector2::new(u as f32, v as f32));

            min_height = min_height.min(s.height);
            max_height = max_height.max(s.height);
        }
    }

    let stride = width_segments + 1;
    for iy in 0..length_segments {
        for ix in 0..width_segments {
            let a = (iy * stride + ix) as u32;
            let b = a + 1;
            let d = ((iy + 1) * stride + ix) as u32;
            let c = d + 1;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }

    MeshData { positions, indices, normals, uvs, min_height, max_height }
}

/// Largest deviation between the mesh's opposite edges, comparing each edge
/// vertex with its counterpart shifted by one tile. Measures x, y, z and the
/// normal, so choppy displacement is covered too.
fn max_seam_error(
    mesh: &MeshData,
    width: f32,
    length: f32,
    width_segments: usize,
    length_segments: usize,
) -> f32 {
    let stride = width_segments + 1;
    let mut max_err: f32 = 0.0;
    let mut compare = |a: usize, b: usize, shift_x: f32, shift_y: f32| {
        let (pa, pb) = (&mesh.positions[a], &mesh.positions[b]);
        let (na, nb) = (&mesh.normals[a], &mesh.normals[b]);
        for e in [
            pb.x - shift_x - pa.x,
            pb.y - shift_y - pa.y,
            pb.z - pa.z,
            nb.x - na.x,
            nb.y - na.y,
            nb.z - na.z,
        ] {
            max_err = max_err.max(e.abs());
        }
    };
    for iy in 0..=length_segments {
        compare(iy * stride, iy * stride + width_segments, width, 0.0);
    }
    for ix in 0..=width_segments {
        compare(ix, length_segments * stride + ix, 0.0, length);
    }
    max_err
}

/// Export the mesh to a GLB file with an NxN tile pattern in the xy-plane
fn export_mesh(
    mesh: &MeshData,
    width: f32,
    length: f32,
    output_path: &PathBuf,
    tiles: usize,
) -> Result<()> {
    let (positions, indices, normals, uvs) =
        (&mesh.positions, &mesh.indices, &mesh.normals, &mesh.uvs);
    use mesh_tools::{GltfBuilder, Triangle};
    
    // Create a new GLTF builder
    let mut builder = GltfBuilder::new();
    
    // Create a single blue material for the ocean
    let ocean_material = builder.create_basic_material(
        Some("OceanMaterial".to_string()),
        [0.0, 0.3, 0.8, 1.0]  // Standard blue
    );
    
    // Convert indices to triangles
    let mut triangles = Vec::with_capacity(indices.len() / 3);
    for i in (0..indices.len()).step_by(3) {
        if i + 2 < indices.len() {
            triangles.push(Triangle::new(
                indices[i], 
                indices[i+1], 
                indices[i+2]
            ));
        }
    }
    
    // Create the mesh using the public API - we'll reuse this mesh for all tiles
    let mesh_index = builder.create_simple_mesh(
        Some("OceanSurface".to_string()),
        positions,
        &triangles,
        Some(normals.to_vec()),
        Some(uvs.to_vec()),
        Some(ocean_material) // Single material for all tiles
    );
    
    // Tiles are spaced by the nominal period, not the displaced bounding box.
    println!("Ocean tile dimensions: width={:.2}, length={:.2}", width, length);

    // Create an NxN grid of nodes using the same mesh but with different transforms
    let mut node_indices = Vec::new();
    
    // Calculate tile placement ranges based on tiles parameter
    let half_tiles = tiles as isize / 2;
    let start = -half_tiles;
    let end = if tiles % 2 == 0 { half_tiles - 1 } else { half_tiles };
    
    for row in start..=end {
        for col in start..=end {
            // Create a transform matrix to position each tile
            let translate_x = col as f32 * width;
            let translate_y = row as f32 * length;
            
            // Determine if this is the center tile
            let is_center = (tiles % 2 == 1) && (row == 0 && col == 0);
            
            // For center tile, use the mesh we already created
            if is_center {
                let node_index = builder.add_node(
                    Some(format!("OceanTile_{}_{}", row, col)),
                    Some(mesh_index),
                    None, // No transform for center tile
                    None, 
                    None
                );
                node_indices.push(node_index);
            } else {
                // For other tiles, create a new mesh with the same material
                let tile_mesh_index = builder.create_simple_mesh(
                    Some(format!("OceanSurface_{}_{}", row, col)),
                    positions,
                    &triangles,
                    Some(normals.to_vec()),
                    Some(uvs.to_vec()),
                    Some(ocean_material)
                );
                
                // Create a translation vector [x, y, z]
                let translation = [translate_x, translate_y, 0.0];
                
                let node_index = builder.add_node(
                    Some(format!("OceanTile_{}_{}", row, col)),
                    Some(tile_mesh_index),
                    Some(translation),
                    None, 
                    None
                );
                node_indices.push(node_index);
            }
        }
    }
    
    // Add a scene with all our ocean tile nodes
    builder.add_scene(Some("Ocean Tiles Scene".to_string()), Some(node_indices));
    
    // Export to GLB
    builder.export_glb(output_path.to_str().unwrap())?;
    
    println!("Mesh exported to: {} ({}x{} tiled pattern)", output_path.display(), tiles, tiles);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(w: f32, l: f32, ws: usize, ls: usize) -> MeshData {
        let size = w.max(l);
        let params = tessendorf::OceanParams {
            resolution: ws.max(ls),
            size,
            choppiness: 0.8,
            ..Default::default()
        };
        let field = tessendorf::compute_wave_field(&params, 0.7);
        create_mesh(w, l, ws, ls, size, &field)
    }

    #[test]
    fn square_mesh_is_seamless() {
        let m = build(20.0, 20.0, 64, 64);
        assert!(max_seam_error(&m, 20.0, 20.0, 64, 64) <= SEAM_TOLERANCE);
    }

    #[test]
    fn seamless_when_segments_differ_from_field_resolution() {
        // x=48 segments, y=32 segments -> field resolution 48, y is interpolated.
        let m = build(20.0, 20.0, 48, 32);
        assert!(max_seam_error(&m, 20.0, 20.0, 48, 32) <= SEAM_TOLERANCE);
    }

    #[test]
    fn seam_check_detects_a_broken_edge() {
        let mut m = build(20.0, 20.0, 16, 16);
        m.positions[16].z += 0.5; // right edge of first row
        assert!(max_seam_error(&m, 20.0, 20.0, 16, 16) > SEAM_TOLERANCE);
    }

    #[test]
    fn edge_vertices_are_not_flattened() {
        // Regression: old code forced edge normals to (0,0,1).
        let m = build(20.0, 20.0, 64, 64);
        assert!(m.normals[..65].iter().any(|n| n.x.abs() > 1e-3 || n.y.abs() > 1e-3));
    }
}
