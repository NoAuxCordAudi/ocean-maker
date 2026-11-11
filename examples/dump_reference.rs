//! Dumps a small ocean's spectrum and Jacobian as JSON so an independent implementation
//! (tests/ref/foam_ref.py) can be compared against it. Usage: dump_reference <out.json>

use std::fs::File;

use ocean_generator::tessendorf::{OceanParams, OceanSim};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).ok_or("usage: dump_reference <out.json>")?;
    let (n, size, time) = (32usize, 20.0f32, 0.7f32);
    let params = OceanParams { resolution: n, size, choppiness: 1.3, amplitude: 1.0, seed: 7, ..OceanParams::default() };
    let sim = OceanSim::new(&params)?;
    let scale = sim.height_scale(&[time])?;
    let jac = sim.jacobians(&[time], scale, n, 0.0)?.remove(0);
    let heights = sim.fields(&[time], scale)?.remove(0).height;

    let spectrum: Vec<Vec<[f32; 2]>> =
        sim.spectrum(time).iter().map(|row| row.iter().map(|c| [c.re, c.im]).collect()).collect();
    let json = serde_json::json!({
        "n": n, "size": size, "choppiness": params.choppiness, "scale": scale,
        "spectrum": spectrum, "height": heights,
        "lambda_min": jac.lambda_min, "det": jac.det,
    });
    serde_json::to_writer(File::create(out)?, &json)?;
    Ok(())
}
