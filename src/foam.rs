//! Whitecap foam: a per-texel mask in [0, 1] for each frame.
//!
//! Model (see `docs/whitecaps-proposal.md`):
//! 1. **Trigger** `g = saturate((lambda_T - lambda_min) / (softness * (1 - lambda_T)))`, where
//!    `lambda_min` is the smaller eigenvalue of the Jacobian of the choppy displacement map
//!    (< 1 where the surface compresses, <= 0 where it folds). The ramp is relative to
//!    `1 - lambda_T` so it scales with how far the threshold is from "undistorted". It is a
//!    demo heuristic, not part of Tessendorf's method.
//! 2. **Decay** `F_{n+1} = max(F_n * exp(-dt / tau), g_{n+1})`, pointwise on the undisplaced
//!    (Lagrangian) grid, so the foam is periodic and tiles seamlessly. `tau` is an e-folding
//!    time. There is no advection: foam stays at fixed material points and leaves trails.
//! 3. **Spin-up**: frames before the clip start are simulated and discarded so the clip does
//!    not start with empty foam.
//! 4. **Calibration**: `lambda_T` is the `coverage` quantile of `lambda_min` pooled over all
//!    simulated frames (spin-up included), so `coverage` is the per-frame fraction of texels
//!    that trigger. Steady-state foam coverage is higher because of the decay.

use anyhow::{ensure, Result};

use crate::tessendorf::OceanSim;

/// How the trigger threshold `lambda_T` is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Trigger {
    /// Pick `lambda_T` so this fraction (0..1, exclusive) of texel-frames trigger.
    Coverage(f32),
    /// Use this `lambda_T` directly.
    Threshold(f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoamParams {
    pub trigger: Trigger,
    /// Ramp width as a fraction of `1 - lambda_T`; 0 would be a hard threshold.
    pub softness: f32,
    /// Foam e-folding time in seconds.
    pub tau: f32,
    /// Spin-up duration in seconds; `None` = automatic (4.6 tau, i.e. 1% residual).
    pub spinup: Option<f32>,
    /// Foam texture resolution (texels per side); at least the simulation resolution.
    pub resolution: usize,
}

impl Default for FoamParams {
    fn default() -> Self {
        Self { trigger: Trigger::Coverage(0.02), softness: 0.15, tau: 2.0, spinup: None, resolution: 256 }
    }
}

/// Foam frames for the clip, plus how they were calibrated.
#[derive(Debug, Clone)]
pub struct FoamSequence {
    pub resolution: usize,
    /// Threshold actually used.
    pub threshold: f32,
    /// Spin-up frames simulated (and discarded).
    pub spinup_frames: usize,
    /// Upper bound of the weight of the un-simulated past at frame 0 (`alpha^spinup_frames`).
    pub residual: f32,
    /// Fraction of simulated texel-frames with `lambda_min < threshold`.
    pub trigger_fraction: f32,
    /// Mean foam value over the clip frames.
    pub mean_foam: f32,
    /// One mask per clip frame, row-major: `index = row * resolution + col`, where `col` is the
    /// x index and `row` the y index. Texel `(col, row)` is centred at
    /// `((col + 0.5) / M, (row + 0.5) / M)` of the patch, so row `r` of an image written in this
    /// order is glTF `v = (r + 0.5) / M`.
    pub frames: Vec<Vec<f32>>,
}

/// Whitecap fraction of the sea surface from 10 m wind speed (m/s): Monahan & O'Muircheartaigh
/// (1980), `W = 3.84e-6 * U^3.41` as a **fraction** (not percent). Parameterisations differ by
/// more than 10x at low wind; at 5 m/s this gives only ~0.09%.
pub fn mom80(wind_speed: f32) -> f32 {
    3.84e-6 * wind_speed.max(0.0).powf(3.41)
}

/// One decay/trigger update of a foam texel.
pub fn step(foam: f32, trigger: f32, alpha: f32) -> f32 {
    (foam * alpha).max(trigger)
}

fn trigger_value(lambda: f32, threshold: f32, softness: f32) -> f32 {
    ((threshold - lambda) / (softness * (1.0 - threshold))).clamp(0.0, 1.0)
}

/// The `coverage` quantile of `values`: the threshold below which that fraction of values lie.
///
/// # Errors
/// If `values` is empty or contains a non-finite value, or `coverage` is not in (0, 1).
pub fn calibrate_threshold(values: &mut [f32], coverage: f32) -> Result<f32> {
    ensure!(coverage > 0.0 && coverage < 1.0, "foam coverage must be in (0, 1), got {coverage}");
    ensure!(!values.is_empty(), "no values to calibrate against");
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite lambda_min encountered");
    let idx = ((values.len() as f64 * coverage as f64) as usize).min(values.len() - 1);
    let (_, q, _) = values.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    Ok(*q)
}

/// Simulate foam for the clip frames `clip_times` (uniformly spaced by `dt`).
///
/// `scale` must be the same height scale used for the mesh (see [`OceanSim::height_scale`]).
///
/// # Errors
/// - invalid parameters (`dt`, `tau`, `softness`, resolution, coverage);
/// - choppiness 0 (no folding is possible);
/// - the calibrated threshold is so close to 1 that "foam" would just be undistorted surface
///   (`lambda_T >= 0.98`): raise choppiness or amplitude;
/// - the resulting foam is identically zero.
pub fn simulate_foam(
    sim: &OceanSim,
    scale: f32,
    clip_times: &[f32],
    dt: f32,
    params: &FoamParams,
) -> Result<FoamSequence> {
    ensure!(!clip_times.is_empty(), "at least one clip time is required");
    ensure!(dt.is_finite() && dt > 0.0, "foam dt must be > 0, got {dt}");
    ensure!(params.tau.is_finite() && params.tau > 0.0, "foam tau must be > 0, got {}", params.tau);
    ensure!(
        params.softness.is_finite() && params.softness > 0.0,
        "foam softness must be > 0, got {}",
        params.softness
    );
    ensure!(
        params.resolution >= sim.resolution().max(2),
        "foam resolution {} must be >= simulation resolution {}",
        params.resolution,
        sim.resolution()
    );
    ensure!(sim.choppiness() > 0.0, "foam needs choppiness > 0 (no surface folding otherwise)");

    let alpha = (-dt / params.tau).exp();
    let auto_frames = ((0.01f32).ln() / alpha.ln()).ceil() as usize;
    let spinup_frames = match params.spinup {
        None => auto_frames,
        Some(s) => {
            ensure!(s.is_finite() && s >= 0.0, "foam spin-up must be >= 0, got {s}");
            (s / dt).ceil() as usize
        }
    };
    let residual = alpha.powi(spinup_frames as i32);

    // Spin-up frames precede the clip at the same spacing; they are simulated and dropped.
    let start = clip_times[0];
    let all_times: Vec<f32> = (0..spinup_frames)
        .map(|i| start - dt * (spinup_frames - i) as f32)
        .chain(clip_times.iter().copied())
        .collect();

    // lambda_min for every frame, row-major (x fastest), sampled at texel centres.
    let m = params.resolution;
    let mut lambdas: Vec<Vec<f32>> = Vec::with_capacity(all_times.len());
    for chunk in all_times.chunks(16) {
        for j in sim.jacobians(chunk, scale, m, 0.5)? {
            let mut flat = Vec::with_capacity(m * m);
            for row in 0..m {
                for col in 0..m {
                    flat.push(j.lambda_min[col][row]);
                }
            }
            lambdas.push(flat);
        }
    }

    let threshold = match params.trigger {
        Trigger::Threshold(t) => {
            ensure!(t.is_finite(), "foam threshold must be finite");
            t
        }
        Trigger::Coverage(c) => {
            let mut pooled: Vec<f32> = lambdas.iter().flatten().copied().collect();
            calibrate_threshold(&mut pooled, c)?
        }
    };
    ensure!(
        threshold < 0.98,
        "foam threshold lambda_T = {threshold:.3} is too close to 1: the surface barely compresses, \
         so foam would sit on non-breaking crests. Raise --choppiness or --amplitude."
    );

    let total = lambdas.len() * m * m;
    let triggered = lambdas.iter().flatten().filter(|&&l| l < threshold).count();
    let mut foam = vec![0.0f32; m * m];
    let mut frames = Vec::with_capacity(clip_times.len());
    for (n, lambda) in lambdas.iter().enumerate() {
        for (f, &l) in foam.iter_mut().zip(lambda) {
            *f = step(*f, trigger_value(l, threshold, params.softness), alpha);
        }
        if n >= spinup_frames {
            frames.push(foam.clone());
        }
    }

    let mean_foam = frames.iter().flatten().sum::<f32>() / (frames.len() * m * m) as f32;
    ensure!(
        mean_foam > 0.0,
        "foam is identically zero for these parameters (threshold {threshold:.3}); \
         raise --choppiness or --foam-coverage"
    );

    Ok(FoamSequence {
        resolution: m,
        threshold,
        spinup_frames,
        residual,
        trigger_fraction: triggered as f32 / total as f32,
        mean_foam,
        frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tessendorf::{frame_times, OceanParams};

    fn params(c: f32) -> OceanParams {
        OceanParams { resolution: 32, size: 20.0, choppiness: c, amplitude: 1.0, ..OceanParams::default() }
    }

    fn run(c: f32, fp: &FoamParams, frames: usize) -> Result<FoamSequence> {
        let sim = OceanSim::new(&params(c))?;
        let dt = 1.0 / 12.0;
        let times: Vec<f32> = (0..frames).map(|i| 1.0 + i as f32 * dt).collect();
        let scale = sim.height_scale(&times)?;
        simulate_foam(&sim, scale, &times, dt, fp)
    }

    fn small() -> FoamParams {
        FoamParams { resolution: 32, ..FoamParams::default() }
    }

    #[test]
    fn decay_only_matches_exponential() {
        let (tau, dt) = (2.0f32, 0.05f32);
        let alpha = (-dt / tau).exp();
        let mut f = 0.8f32;
        for _ in 0..40 {
            f = step(f, 0.0, alpha);
        }
        assert!((f - 0.8 * (-(40.0 * dt) / tau).exp()).abs() < 1e-6);
        // two steps of dt == one step of 2 dt
        let two = step(step(0.6, 0.0, alpha), 0.0, alpha);
        let one = step(0.6, 0.0, (-(2.0 * dt) / tau).exp());
        assert!((two - one).abs() < 1e-6);
    }

    #[test]
    fn step_takes_the_larger_of_decay_and_trigger() {
        assert_eq!(step(0.5, 0.9, 0.5), 0.9);
        assert_eq!(step(0.8, 0.1, 0.5), 0.4);
    }

    #[test]
    fn more_spinup_changes_frame_zero_by_at_most_the_residual() {
        let base = FoamParams { trigger: Trigger::Threshold(0.8), ..small() };
        let short = run(1.0, &FoamParams { spinup: Some(1.0), ..base }, 3).unwrap();
        let long = run(1.0, &FoamParams { spinup: Some(12.0), ..base }, 3).unwrap();
        let max_diff = short.frames[0]
            .iter()
            .zip(&long.frames[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff <= short.residual + 1e-6, "{max_diff} > residual {}", short.residual);
        assert!(short.residual > long.residual);
    }

    #[test]
    fn auto_spinup_leaves_about_one_percent() {
        let s = run(1.0, &small(), 2).unwrap();
        assert!(s.residual <= 0.0101 && s.residual > 0.005, "{}", s.residual);
        assert!(s.spinup_frames > 0);
    }

    #[test]
    fn realised_trigger_fraction_matches_target() {
        let s = run(1.0, &small(), 6).unwrap();
        assert!((s.trigger_fraction - 0.02).abs() < 0.004, "{}", s.trigger_fraction);
    }

    #[test]
    fn higher_coverage_gives_higher_threshold() {
        let lo = run(1.0, &FoamParams { trigger: Trigger::Coverage(0.01), ..small() }, 3).unwrap();
        let hi = run(1.0, &FoamParams { trigger: Trigger::Coverage(0.10), ..small() }, 3).unwrap();
        assert!(hi.threshold > lo.threshold);
        assert!(hi.mean_foam > lo.mean_foam);
    }

    #[test]
    fn foam_is_bounded_finite_and_deterministic() {
        let a = run(1.2, &small(), 4).unwrap();
        let b = run(1.2, &small(), 4).unwrap();
        assert_eq!(a.frames, b.frames);
        assert_eq!(a.frames.len(), 4);
        assert!(a.frames.iter().flatten().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!(a.frames[0].len() == 32 * 32);
    }

    #[test]
    fn foam_reaches_full_intensity_at_the_peak() {
        let s = run(1.2, &small(), 4).unwrap();
        let max = s.frames.iter().flatten().cloned().fold(0.0f32, f32::max);
        assert!((max - 1.0).abs() < 1e-6, "peak foam {max}");
    }

    #[test]
    fn foam_sits_on_crests() {
        // Fresh foam must sit on crests, not troughs. Use a tiny tau so only the current
        // trigger shows: persistent foam is a trail left *behind* crests that have moved on.
        let p = params(1.5);
        let sim = OceanSim::new(&p).unwrap();
        let times = [1.0f32, 1.0 + 1.0 / 12.0];
        let scale = sim.height_scale(&times).unwrap();
        let fp = FoamParams { trigger: Trigger::Coverage(0.03), tau: 0.001, spinup: Some(0.0), resolution: 64, ..FoamParams::default() };
        let foam = simulate_foam(&sim, scale, &times, 1.0 / 12.0, &fp).unwrap();
        let field = sim.fields(&times[..1], scale).unwrap().remove(0);
        let m = foam.resolution;
        let (mut sum_foamy, mut n_foamy, mut sum_all) = (0.0f64, 0usize, 0.0f64);
        for row in 0..m {
            for col in 0..m {
                let h = field
                    .sample_periodic((col as f64 + 0.5) / m as f64 * 32.0, (row as f64 + 0.5) / m as f64 * 32.0)
                    .height as f64;
                sum_all += h;
                if foam.frames[0][row * m + col] > 0.5 {
                    sum_foamy += h;
                    n_foamy += 1;
                }
            }
        }
        assert!(n_foamy > 10, "too little foam to test: {n_foamy}");
        let mean_foamy = sum_foamy / n_foamy as f64;
        let mean_all = sum_all / (m * m) as f64;
        assert!(mean_foamy > mean_all + 0.1, "foam height {mean_foamy} vs mean {mean_all}");
    }

    #[test]
    fn fails_fast_when_foam_is_unreachable() {
        // no choppiness at all
        assert!(run(0.0, &small(), 2).is_err());
        // nearly flat displacement: threshold would sit at ~1
        let err = run(0.001, &small(), 2).unwrap_err().to_string();
        assert!(err.contains("Raise --choppiness"), "{err}");
        // explicit threshold below every texel: no foam at all
        let none = FoamParams { trigger: Trigger::Threshold(-50.0), ..small() };
        assert!(run(1.0, &none, 2).is_err());
    }

    #[test]
    fn rejects_bad_parameters() {
        let ok = small();
        assert!(run(1.0, &FoamParams { tau: 0.0, ..ok }, 2).is_err());
        assert!(run(1.0, &FoamParams { softness: 0.0, ..ok }, 2).is_err());
        assert!(run(1.0, &FoamParams { resolution: 16, ..ok }, 2).is_err());
        assert!(run(1.0, &FoamParams { trigger: Trigger::Coverage(0.0), ..ok }, 2).is_err());
        assert!(run(1.0, &FoamParams { trigger: Trigger::Coverage(1.0), ..ok }, 2).is_err());
        assert!(run(1.0, &FoamParams { spinup: Some(-1.0), ..ok }, 2).is_err());
        let sim = OceanSim::new(&params(1.0)).unwrap();
        assert!(simulate_foam(&sim, 1.0, &[], 0.1, &ok).is_err());
        assert!(simulate_foam(&sim, 1.0, &[0.0], 0.0, &ok).is_err());
    }

    #[test]
    fn calibration_returns_the_quantile() {
        let mut v: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(calibrate_threshold(&mut v, 0.1).unwrap(), 100.0);
        assert!(calibrate_threshold(&mut [], 0.1).is_err());
        assert!(calibrate_threshold(&mut [f32::NAN], 0.1).is_err());
    }

    #[test]
    fn mom80_units_are_a_fraction() {
        // 3.84e-6 is a fraction: 0.093% at 5 m/s, 0.99% at 10, 3.9% at 15.
        assert!((mom80(5.0) - 9.3e-4).abs() < 1e-5);
        assert!((mom80(10.0) - 9.87e-3).abs() < 1e-4);
        assert!((mom80(15.0) - 0.0392).abs() < 5e-4);
        assert_eq!(mom80(0.0), 0.0);
    }

    #[test]
    fn frame_times_feed_the_simulation() {
        let times = frame_times(0.0, 0.5, 8.0).unwrap();
        let sim = OceanSim::new(&params(1.0)).unwrap();
        let scale = sim.height_scale(&times).unwrap();
        let s = simulate_foam(&sim, scale, &times, 1.0 / 8.0, &small()).unwrap();
        assert_eq!(s.frames.len(), times.len());
    }
}
