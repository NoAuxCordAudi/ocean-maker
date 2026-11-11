//! Writes a sequence of OBJ frames of an evolving ocean to `./animation_output`.

mod common;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

use ocean_generator::tessendorf::{compute_wave_field, OceanParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 20.0;
    let params = OceanParams {
        resolution: 64,
        size,
        wind_speed: 5.0,
        amplitude: 0.7,
        choppiness: 0.6,
        ..OceanParams::default()
    };

    let frame_count = 60;
    let time_step = 0.05;
    let output_dir = PathBuf::from("./animation_output");
    std::fs::create_dir_all(&output_dir)?;

    println!("Generating {frame_count} frames...");
    let start = Instant::now();
    for frame in 0..frame_count {
        let field = compute_wave_field(&params, frame as f32 * time_step);
        let mesh = common::ocean_mesh(&field, size);
        let path = output_dir.join(format!("ocean_frame_{frame:03}.obj"));
        common::write_obj(&mesh, &mut BufWriter::new(File::create(&path)?))?;
    }
    println!("Done in {:.2?}; frames in {}", start.elapsed(), output_dir.display());
    Ok(())
}
