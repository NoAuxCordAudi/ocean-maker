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

    /// Wind direction X component
    #[arg(long, default_value_t = 1.0, allow_negative_numbers = true)]
    wind_dir_x: f32,

    /// Wind direction Y component
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    wind_dir_y: f32,

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

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    println!("Generating Tessendorf ocean surface...");
    println!("Ocean dimensions: {}x{} with {}x{} segments", 
             args.width, args.length, args.width_segments, args.length_segments);
    println!("Wave parameters: wind speed={}, amplitude={}, choppiness={}, time={}", 
             args.wind_speed, args.amplitude, args.choppiness, args.time);

    // Create ocean parameters
    let params = tessendorf::OceanParams {
        resolution: args.width_segments.max(args.length_segments),
        size: args.width.max(args.length),
        wind_speed: args.wind_speed,
        wind_direction: glam::Vec2::new(args.wind_dir_x, args.wind_dir_y).normalize(),
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
    let (positions, indices, normals, uvs, min_height, max_height) = create_mesh(
        args.width,
        args.length,
        args.width_segments,
        args.length_segments,
        &displacement,
    );
    
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
    export_mesh(&positions, &indices, &normals, &uvs, &args.output, args.tiles)?;

    println!("Done!");
    Ok(())
}

/// Create a mesh from the displacement field
/// Note: Ocean surface is defined in the xy-plane, with z-axis as height
fn create_mesh(
    width: f32,
    length: f32,
    width_segments: usize,
    length_segments: usize,
    displacement: &tessendorf::DisplacementField,
) -> (Vec<Point3<f32>>, Vec<u32>, Vec<Vector3<f32>>, Vec<Vector2<f32>>, f32, f32) {
    let grid_x = width_segments;
    let grid_z = length_segments;
    let resolution = displacement.resolution;

    // Calculate grid size
    let segment_width = width / grid_x as f32;
    let segment_length = length / grid_z as f32;

    let half_width = width / 2.0;
    let half_length = length / 2.0;

    // Prepare vertex arrays
    let mut positions = Vec::with_capacity((grid_x + 1) * (grid_z + 1));
    let mut indices = Vec::with_capacity(grid_x * grid_z * 6);
    let mut normals = Vec::with_capacity((grid_x + 1) * (grid_z + 1));
    let mut uvs = Vec::with_capacity((grid_x + 1) * (grid_z + 1));

    // Track min/max height values
    let mut min_height: f32 = f32::MAX;
    let mut max_height: f32 = f32::MIN;

    // Generate vertices
    for iy in 0..=grid_z { // Using grid_z for y-segments since we're repurposing the parameters
        let y = iy as f32 * segment_length - half_length;
        
        for ix in 0..=grid_x {
            let x = ix as f32 * segment_width - half_width;
            
            // Sample displacement field at this point
            // Map from model coordinates to displacement field coordinates
            let sample_x = ((ix as f32 / grid_x as f32) * (resolution - 1) as f32).round() as usize;
            let sample_y = ((iy as f32 / grid_z as f32) * (resolution - 1) as f32).round() as usize;
            let sample_x = sample_x.min(resolution - 1);
            let sample_y = sample_y.min(resolution - 1);
            
            // Apply displacements from the wave field
            let dx = displacement.displacement_x[sample_x][sample_y];
            let dy = displacement.displacement_y[sample_x][sample_y];
            
            // Get the wave height from displacement field
            let vertex_z = displacement.height[sample_x][sample_y];
            
            // Add position with displacement
            positions.push(mesh_tools::compat::point3::new(x + dx, y + dy, vertex_z));
            
            // Track min/max height
            min_height = min_height.min(vertex_z);
            max_height = max_height.max(vertex_z);
            
            // Add normal
            // Remap normals for xy-plane with z as up direction
            let nx = displacement.normal_x[sample_x][sample_y];
            let ny = displacement.normal_y[sample_x][sample_y];
            let nz = displacement.normal_z[sample_x][sample_y];
            normals.push(mesh_tools::compat::vector3::new(nx, ny, nz));
            
            // Add texture coordinate
            uvs.push(mesh_tools::compat::vector2::new(
                ix as f32 / grid_x as f32,
                iy as f32 / grid_z as f32,
            ));
        }
    }

    // Generate indices for triangles
    for iy in 0..grid_z {
        for ix in 0..grid_x {
            let a = (iy * (grid_x + 1) + ix) as u32;
            let b = (iy * (grid_x + 1) + ix + 1) as u32;
            let c = ((iy + 1) * (grid_x + 1) + ix + 1) as u32;
            let d = ((iy + 1) * (grid_x + 1) + ix) as u32;
            
            // Two triangles per grid cell
            indices.push(a);
            indices.push(b);
            indices.push(d);
            
            indices.push(b);
            indices.push(c);
            indices.push(d);
        }
    }

    // Enforce periodic boundary conditions to ensure tileable mesh
    // Make opposite edges have identical heights
    for i in 0..=length_segments {
        // Make left and right edges identical
        let left_idx = i * (width_segments + 1);
        let right_idx = left_idx + width_segments;
        
        // Choose the average height of left and right edges
        let avg_height = (positions[left_idx].z + positions[right_idx].z) / 2.0;
        positions[left_idx].z = avg_height;
        positions[right_idx].z = avg_height;
        
        // Also adjust normals to be vertical at edges for better tiling
        normals[left_idx] = mesh_tools::compat::vector3::new(0.0, 0.0, 1.0);
        normals[right_idx] = mesh_tools::compat::vector3::new(0.0, 0.0, 1.0);
    }
    
    for j in 0..=width_segments {
        // Make bottom and top edges identical
        let bottom_idx = j;
        let top_idx = length_segments * (width_segments + 1) + j;
        
        // Choose the average height of bottom and top edges
        let avg_height = (positions[bottom_idx].z + positions[top_idx].z) / 2.0;
        positions[bottom_idx].z = avg_height;
        positions[top_idx].z = avg_height;
        
        // Also adjust normals to be vertical at edges for better tiling
        normals[bottom_idx] = mesh_tools::compat::vector3::new(0.0, 0.0, 1.0);
        normals[top_idx] = mesh_tools::compat::vector3::new(0.0, 0.0, 1.0);
    }
    
    // Create arrays to store edge heights for consistency checking
    let mut left_edge = Vec::with_capacity(length_segments + 1);
    let mut right_edge = Vec::with_capacity(length_segments + 1);
    let mut bottom_edge = Vec::with_capacity(width_segments + 1);
    let mut top_edge = Vec::with_capacity(width_segments + 1);
    
    // Extract heights at the edges
    for i in 0..=length_segments {
        // Left edge (x = -width/2)
        let left_idx = i * (width_segments + 1);
        left_edge.push(positions[left_idx].z);
        
        // Right edge (x = width/2)
        let right_idx = left_idx + width_segments;
        right_edge.push(positions[right_idx].z);
    }
    
    for i in 0..=width_segments {
        // Bottom edge (y = -length/2)
        let bottom_idx = i;
        bottom_edge.push(positions[bottom_idx].z);
        
        // Top edge (y = length/2)
        let top_idx = length_segments * (width_segments + 1) + i;
        top_edge.push(positions[top_idx].z);
    }
    
    // Check if opposite edges are consistent within tolerance
    let tolerance = 0.001;
    let mut max_left_right_diff: f32 = 0.0;
    let mut max_top_bottom_diff: f32 = 0.0;
    
    for i in 0..left_edge.len() {
        let diff = (left_edge[i] - right_edge[i]).abs();
        max_left_right_diff = max_left_right_diff.max(diff);
    }
    
    for i in 0..bottom_edge.len() {
        let diff = (bottom_edge[i] - top_edge[i]).abs();
        max_top_bottom_diff = max_top_bottom_diff.max(diff);
    }
    
    println!("Edge consistency check:");
    println!("  Left-Right edges max difference: {:.6} units", max_left_right_diff);
    println!("  Bottom-Top edges max difference: {:.6} units", max_top_bottom_diff);
    
    if max_left_right_diff <= tolerance && max_top_bottom_diff <= tolerance {
        println!("  ✓ Mesh is consistent across tiles (within tolerance of {:.6})", tolerance);
    } else {
        println!("  ✗ Mesh is NOT consistent across tiles (exceeds tolerance of {:.6})", tolerance);
        println!("    This may cause visible seams when tiling the mesh");
        
        // Print some sample points for debugging
        println!("  Sample edge points (height values):");
        println!("    Left edge [0]: {:.6}, Right edge [0]: {:.6}", left_edge[0], right_edge[0]);
        println!("    Left edge [mid]: {:.6}, Right edge [mid]: {:.6}", 
                 left_edge[left_edge.len()/2], right_edge[right_edge.len()/2]);
        println!("    Bottom edge [0]: {:.6}, Top edge [0]: {:.6}", bottom_edge[0], top_edge[0]);
        println!("    Bottom edge [mid]: {:.6}, Top edge [mid]: {:.6}", 
                 bottom_edge[bottom_edge.len()/2], top_edge[top_edge.len()/2]);
    }

    // Return the mesh data along with min/max height values
    (positions, indices, normals, uvs, min_height, max_height)
}

/// Export the mesh to a GLB file with an NxN tile pattern in the xy-plane
fn export_mesh(
    positions: &[Point3<f32>],
    indices: &[u32],
    normals: &[Vector3<f32>],
    uvs: &[Vector2<f32>],
    output_path: &PathBuf,
    tiles: usize,
) -> Result<()> {
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
    
    // Find the size of the mesh for proper tiling
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    
    for pos in positions {
        min_x = min_x.min(pos.x);
        max_x = max_x.max(pos.x);
        min_y = min_y.min(pos.y);
        max_y = max_y.max(pos.y);
    }
    
    let width = max_x - min_x;
    let height = max_y - min_y;
    
    println!("Ocean tile dimensions: width={:.2}, height={:.2}", width, height);
    
    // Create an NxN grid of nodes using the same mesh but with different transforms
    let mut node_indices = Vec::new();
    
    // Calculate tile placement ranges based on tiles parameter
    let half_tiles = tiles as isize / 2;
    let start = if tiles % 2 == 0 { -half_tiles } else { -half_tiles };
    let end = if tiles % 2 == 0 { half_tiles - 1 } else { half_tiles };
    
    for row in start..=end {
        for col in start..=end {
            // Create a transform matrix to position each tile
            let translate_x = col as f32 * width;
            let translate_y = row as f32 * height;
            
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
