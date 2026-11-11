use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use ocean_generator::export::{export_glb, morph_data_bytes, write_foam_sequence, Animation, FoamExport};
use ocean_generator::foam::{mom80, simulate_foam, FoamParams, Trigger};
use ocean_generator::mesh::{flat_grid, max_seam_error, ocean_mesh, GridSpec, MeshData, SEAM_TOLERANCE};
use ocean_generator::tessendorf;

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

    /// Simulation time in seconds for a single still frame
    #[arg(short, long, default_value_t = 0.0, conflicts_with = "anim_duration")]
    time: f32,

    /// Export a morph-target animation lasting this many seconds (omit for a still frame)
    #[arg(long)]
    anim_duration: Option<f32>,

    /// Simulation time at which the animation starts
    #[arg(long, default_value_t = 0.0, requires = "anim_duration", allow_negative_numbers = true)]
    anim_start: f32,

    /// Animation sample rate in frames per second
    #[arg(long, default_value_t = 24.0, requires = "anim_duration")]
    anim_fps: f32,

    /// Output file path
    #[arg(short, long, default_value = "ocean_surface.glb")]
    output: PathBuf,

    /// Number of tiles in each direction (creates NxN grid)
    #[arg(short = 'g', long, default_value_t = 1)]
    tiles: usize,

    /// Generate whitecap foam: a 16-bit PNG per frame, foam.json and a Blender setup script
    /// (written to --foam-dir). Needs choppiness > 0 and width == length.
    #[arg(long)]
    foam: bool,

    /// Fraction of the surface that triggers foam in each frame (0..1), or `physical` to use the
    /// Monahan & O'Muircheartaigh (1980) whitecap fraction for --wind-speed (only ~0.09% at 5 m/s)
    #[arg(long, default_value = "0.02", requires = "foam", conflicts_with = "foam_threshold")]
    foam_coverage: CoverageArg,

    /// Foam updates per frame (1-16); raise if the foam looks temporally aliased
    #[arg(long, default_value_t = 1, requires = "foam")]
    foam_substeps: usize,

    /// Break the foam mask up with procedural noise in Blender (written to the manifest)
    #[arg(long, requires = "foam")]
    foam_noise: bool,

    /// Use this Jacobian threshold (lambda_T) instead of calibrating to --foam-coverage
    #[arg(long, requires = "foam", allow_negative_numbers = true)]
    foam_threshold: Option<f32>,

    /// Foam edge softness, as a fraction of (1 - lambda_T)
    #[arg(long, default_value_t = 0.15, requires = "foam")]
    foam_softness: f32,

    /// Foam e-folding decay time in seconds
    #[arg(long, default_value_t = 2.0, requires = "foam")]
    foam_tau: f32,

    /// Foam spin-up before the first frame, in seconds (default: 4.6 x tau)
    #[arg(long, requires = "foam")]
    foam_spinup: Option<f32>,

    /// Foam image resolution in texels per side (default: max(256, simulation resolution))
    #[arg(long, requires = "foam")]
    foam_resolution: Option<usize>,

    /// Directory for the foam images, manifest and Blender script (default: <output>_foam)
    #[arg(long, requires = "foam")]
    foam_dir: Option<PathBuf>,

    /// Do not write ocean_blender.py
    #[arg(long, requires = "foam")]
    no_blender_script: bool,
}

/// `--foam-coverage`: a fraction in (0, 1) or the literal `physical`.
#[derive(Debug, Clone, Copy)]
enum CoverageArg {
    Fraction(f32),
    Physical,
}

impl std::str::FromStr for CoverageArg {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        if s == "physical" {
            return Ok(Self::Physical);
        }
        s.parse::<f32>()
            .map(Self::Fraction)
            .map_err(|_| format!("expected a fraction like 0.02 or `physical`, got `{s}`"))
    }
}

/// Validate foam options up front (before any expensive work) and build the foam parameters.
fn foam_params(args: &Args, sim_resolution: usize, size: f32) -> Result<FoamParams> {
    ensure!(
        args.width == size && args.length == size,
        "--foam needs width == length: the foam image is mapped onto the mesh UVs, which only \
         covers one full periodic patch when the mesh is square"
    );
    ensure!(args.choppiness > 0.0, "--foam needs --choppiness > 0 (no surface folding otherwise)");
    let defaults = FoamParams::default();
    Ok(FoamParams {
        trigger: match (args.foam_threshold, args.foam_coverage) {
            (Some(t), _) => Trigger::Threshold(t),
            (None, CoverageArg::Fraction(c)) => Trigger::Coverage(c),
            (None, CoverageArg::Physical) => {
                let w = mom80(args.wind_speed);
                println!(
                    "Physical foam coverage for {} m/s wind: {:.3}% (MOM80; parameterisations vary >10x here)",
                    args.wind_speed,
                    w * 100.0
                );
                Trigger::Coverage(w)
            }
        },
        substeps: args.foam_substeps,
        softness: args.foam_softness,
        tau: args.foam_tau,
        spinup: args.foam_spinup,
        resolution: args.foam_resolution.unwrap_or(defaults.resolution.max(sim_resolution)),
    })
}

fn default_foam_dir(output: &Path) -> PathBuf {
    let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("ocean");
    output.with_file_name(format!("{stem}_foam"))
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Generating Tessendorf ocean surface...");
    println!(
        "Ocean dimensions: {}x{} with {}x{} segments",
        args.width, args.length, args.width_segments, args.length_segments
    );
    println!(
        "Wave parameters: wind speed={}, amplitude={}, choppiness={}",
        args.wind_speed, args.amplitude, args.choppiness
    );

    // --wind-angle wins if provided, else x/y components.
    let wind_direction = match args.wind_angle {
        Some(deg) => {
            let rad = deg.to_radians();
            glam::Vec2::new(rad.cos(), rad.sin())
        }
        None => {
            let v = glam::Vec2::new(args.wind_dir_x, args.wind_dir_y);
            ensure!(v.length() > 0.0, "wind direction vector must be non-zero");
            v.normalize()
        }
    };
    println!("Wind direction: ({:.3}, {:.3})", wind_direction.x, wind_direction.y);

    let grid = GridSpec::new(args.width, args.length, args.width_segments, args.length_segments)?;
    ensure!(args.tiles >= 1, "tiles must be at least 1");

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

    // Simulation times: one still frame, or an animation's keyframes.
    let times = match args.anim_duration {
        Some(duration) => {
            let times = tessendorf::frame_times(args.anim_start, duration, args.anim_fps)?;
            let vertices = (args.width_segments + 1) * (args.length_segments + 1);
            println!(
                "Animation: {} frames over {duration}s at {} fps from t={} (~{:.1} MB of morph data)",
                times.len(),
                args.anim_fps,
                args.anim_start,
                morph_data_bytes(vertices, times.len()) as f64 / (1024.0 * 1024.0)
            );
            times
        }
        None => {
            println!("Time: {}", args.time);
            vec![args.time]
        }
    };

    // Validate foam options before doing any expensive work.
    let foam = args.foam.then(|| foam_params(&args, params.resolution, params.size)).transpose()?;

    println!("Computing wave field...");
    let sim = tessendorf::OceanSim::new(&params)?;
    let scale = sim.height_scale(&times)?;
    println!("Height scale: {scale:.6} (peak-to-peak target {:.3})", 2.0 * params.amplitude);
    let fields = sim.fields(&times, scale)?;

    println!("Creating mesh...");
    let meshes: Vec<MeshData> = fields.iter().map(|f| ocean_mesh(&grid, params.size, f)).collect();

    report_amplitude(&meshes, args.amplitude);
    check_seams(&meshes, &grid, &params, args.tiles)?;

    println!("Exporting to {} ({}x{} tiles)...", args.output.display(), args.tiles, args.tiles);
    match args.anim_duration {
        Some(_) => {
            let rest_pose = flat_grid(&grid);
            let animation = Animation { fps: args.anim_fps, frames: &meshes };
            export_glb(&rest_pose, Some(animation), &grid, args.tiles, &args.output)?;
        }
        None => export_glb(&meshes[0], None, &grid, args.tiles, &args.output)?,
    }

    if let Some(fp) = foam {
        let dt = 1.0 / args.anim_fps;
        println!("Simulating foam ({} texel resolution, tau {}s)...", fp.resolution, fp.tau);
        let seq = simulate_foam(&sim, scale, &times, dt, &fp)?;
        println!(
            "Foam: threshold {:.3}, {:.2}% of texel-frames trigger, mean foam {:.3}, \
             {} spin-up frames (residual {:.3})",
            seq.threshold,
            seq.trigger_fraction * 100.0,
            seq.mean_foam,
            seq.spinup_frames,
            seq.residual
        );
        let dir = args.foam_dir.clone().unwrap_or_else(|| default_foam_dir(&args.output));
        let shape_keys = if args.anim_duration.is_some() { meshes.len() } else { 0 };
        let export = FoamExport {
            sequence: &seq,
            params: &fp,
            fps: args.anim_fps,
            glb_path: &args.output,
            shape_keys,
            write_blender_script: !args.no_blender_script,
            noise_breakup: args.foam_noise,
        };
        write_foam_sequence(&dir, &export)
            .with_context(|| format!("failed to write foam files to {}", dir.display()))?;
        println!("Foam files written to {}", dir.display());
        if !args.no_blender_script {
            println!("In Blender: blender --python {}", dir.join("ocean_blender.py").display());
        }
    }

    println!("Done!");
    Ok(())
}

/// Compare the realised height range over all frames with the commanded amplitude.
fn report_amplitude(meshes: &[MeshData], amplitude: f32) {
    let (min_h, max_h) = meshes.iter().map(MeshData::height_range).fold(
        (f32::MAX, f32::MIN),
        |(lo, hi), (l, h)| (lo.min(l), hi.max(h)),
    );
    let range = max_h - min_h;
    let expected = amplitude * 2.0;
    println!("Z-axis displacement range: {min_h:.6} to {max_h:.6} units");
    println!("Extent over all frames: {range:.6} units (expected {expected:.6}, ratio {:.4})", range / expected);
}

/// The mesh is one exact period of the field only when it spans the full field size;
/// in that case every frame must be seamless or we fail.
fn check_seams(meshes: &[MeshData], grid: &GridSpec, params: &tessendorf::OceanParams, tiles: usize) -> Result<()> {
    let periodic = grid.width() == params.size && grid.length() == params.size;
    if periodic {
        let worst = meshes.iter().map(|m| max_seam_error(m, grid)).fold(0.0f32, f32::max);
        println!("Seam error across opposite edges: {worst:.2e}");
        ensure!(
            worst <= SEAM_TOLERANCE,
            "mesh is not seamlessly tileable: seam error {worst:.2e} exceeds {SEAM_TOLERANCE:.2e}"
        );
    } else if tiles > 1 {
        eprintln!(
            "warning: width ({}) != length ({}), so the mesh is only a window of a {size}x{size} \
             periodic field and will show seams when tiled. Use equal width and length for seamless tiling.",
            grid.width(), grid.length(), size = params.size
        );
    }
    Ok(())
}
