//! Times wave-field computation, mesh construction and OBJ serialisation.

mod common;

use std::time::Instant;

use ocean_generator::tessendorf::{compute_wave_field, OceanParams};

fn main() {
    println!("Ocean generator benchmark\n=========================");
    for resolution in [128, 256, 512] {
        let params = OceanParams { resolution, size: 100.0, ..OceanParams::default() };
        println!("\nResolution {resolution}x{resolution} ({} vertices)", (resolution + 1).pow(2));

        let t = Instant::now();
        let field = compute_wave_field(&params, 0.0);
        println!("  Wave field:   {:.2?}", t.elapsed());

        let t = Instant::now();
        let mesh = common::ocean_mesh(&field, params.size);
        println!("  Mesh build:   {:.2?}", t.elapsed());

        // Skip serialisation of the largest mesh to bound memory use.
        if resolution <= 256 {
            let t = Instant::now();
            let mut buffer = Vec::new();
            common::write_obj(&mesh, &mut buffer).expect("writing to a Vec cannot fail");
            println!("  OBJ export:   {:.2?} ({} MB)", t.elapsed(), buffer.len() / (1024 * 1024));
        }
    }
}
