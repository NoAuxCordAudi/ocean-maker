//! Writes a 3-second animated ocean as a single GLB using morph targets.
//! Open `animated_ocean.glb` in Blender (File > Import > glTF) and press play.

use std::path::Path;
use std::time::Instant;

use ocean_generator::export::{export_glb, Animation};
use ocean_generator::mesh::{flat_grid, ocean_mesh, GridSpec};
use ocean_generator::tessendorf::{compute_wave_fields, frame_times, OceanParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 20.0;
    let segments = 64;
    let params = OceanParams {
        resolution: segments,
        size,
        wind_speed: 5.0,
        amplitude: 0.7,
        choppiness: 0.6,
        ..OceanParams::default()
    };
    let (start, duration, fps) = (0.0, 3.0, 24.0);

    let start_time = Instant::now();
    let times = frame_times(start, duration, fps)?;
    println!("Simulating {} frames...", times.len());
    let fields = compute_wave_fields(&params, &times)?;

    let grid = GridSpec::new(size, size, segments, segments)?;
    let frames: Vec<_> = fields.iter().map(|f| ocean_mesh(&grid, size, f)).collect();

    // The rest pose is a flat grid; every frame becomes a morph target.
    let rest_pose = flat_grid(&grid);
    let output = Path::new("animated_ocean.glb");
    export_glb(&rest_pose, Some(Animation { fps, frames: &frames }), &grid, 1, output)?;

    println!("Wrote {} in {:.2?}", output.display(), start_time.elapsed());
    Ok(())
}
