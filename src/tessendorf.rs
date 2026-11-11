use anyhow::{ensure, Result};
use glam::Vec2;
use num_complex::Complex32;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Parameters for the Tessendorf FFT Ocean algorithm
#[derive(Debug, Clone)]
pub struct OceanParams {
    pub resolution: usize,      // Grid resolution (N x N)
    pub size: f32,              // Physical size of the ocean patch in meters
    pub wind_speed: f32,        // Wind speed
    pub wind_direction: Vec2,   // Wind direction (normalized)
    pub amplitude: f32,         // Overall wave amplitude
    pub choppiness: f32,        // Wave choppiness factor (0-1)
    pub gravity: f32,           // Gravity acceleration (typically 9.8)
    pub depth: f32,             // Water depth (deep water when very large)
    pub seed: u64,              // Random seed for wave generation
}

impl Default for OceanParams {
    fn default() -> Self {
        Self {
            resolution: 128,
            size: 100.0,
            wind_speed: 5.0,
            wind_direction: Vec2::new(1.0, 0.0).normalize(),
            amplitude: 1.0,
            choppiness: 0.5,
            gravity: 9.81,
            depth: 200.0,
            seed: 42,
        }
    }
}

/// Displacement field representing horizontal and vertical displacement of the ocean surface
#[derive(Debug)]
pub struct DisplacementField {
    pub resolution: usize,
    pub height: Vec<Vec<f32>>,           // Height displacement
    pub displacement_x: Vec<Vec<f32>>,   // X displacement (for choppy waves)
    pub displacement_y: Vec<Vec<f32>>,   // Y displacement (for choppy waves)
    pub normal_x: Vec<Vec<f32>>,         // Surface normal X component
    pub normal_y: Vec<Vec<f32>>,         // Surface normal Y component
    pub normal_z: Vec<Vec<f32>>,         // Surface normal Z component
}

/// One interpolated sample of a [`DisplacementField`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldSample {
    pub height: f32,
    pub displacement: Vec2,
    pub normal: [f32; 3],
}

impl DisplacementField {
    /// Sample the field at continuous grid coordinates `(u, v)`, measured in
    /// grid cells. The field is periodic with period `resolution`, so
    /// coordinates wrap: `u == resolution` returns exactly the same sample as
    /// `u == 0`. This is what makes tiled meshes seamless.
    pub fn sample_periodic(&self, u: f64, v: f64) -> FieldSample {
        let n = self.resolution;
        let (i0, i1, fx) = wrap_index(u, n);
        let (j0, j1, fy) = wrap_index(v, n);

        let lerp = |f: &Vec<Vec<f32>>| -> f32 {
            let a = f[i0][j0] + (f[i1][j0] - f[i0][j0]) * fx;
            let b = f[i0][j1] + (f[i1][j1] - f[i0][j1]) * fx;
            a + (b - a) * fy
        };

        let mut normal = [lerp(&self.normal_x), lerp(&self.normal_y), lerp(&self.normal_z)];
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 1e-6 {
            normal.iter_mut().for_each(|c| *c /= len);
        } else {
            normal = [0.0, 0.0, 1.0];
        }

        FieldSample {
            height: lerp(&self.height),
            displacement: Vec2::new(lerp(&self.displacement_x), lerp(&self.displacement_y)),
            normal,
        }
    }
}

/// Wrap a continuous grid coordinate into (lower index, upper index, fraction).
fn wrap_index(u: f64, n: usize) -> (usize, usize, f32) {
    let wrapped = u.rem_euclid(n as f64);
    let base = wrapped.floor();
    let frac = (wrapped - base) as f32;
    let i0 = (base as usize) % n;
    (i0, (i0 + 1) % n, frac)
}

/// Map an FFT bin index in [0, n) to its signed wave-number index.
/// Standard FFT layout: bins [0, n/2) are positive frequencies,
/// bins [n/2, n) are negative frequencies. This lets us feed the
/// spectrum directly into an IFFT without an explicit fftshift.
fn fft_freq(i: usize, n: usize) -> f32 {
    if i < n / 2 {
        i as f32
    } else {
        i as f32 - n as f32
    }
}

/// Phillips spectrum implementation for wave height distribution
/// This determines the statistical distribution of wave heights
fn phillips_spectrum(k: Vec2, params: &OceanParams) -> f32 {
    // If wave vector is zero, return zero
    if k.length_squared() < 1e-6 {
        return 0.0;
    }

    // Largest possible waves arising from wind of speed V
    let l = params.wind_speed * params.wind_speed / params.gravity;
    let k_length = k.length();
    let k_hat = k.normalize();
    
    // Dot product of normalized wave vector and wind direction
    let dot_k_hat_omega_hat = k_hat.dot(params.wind_direction);
    
    // Phillips spectrum formula (see paper by Tessendorf)
    let mut result = params.amplitude * (-1.0 / (k_length * l).powi(2)).exp() / k_length.powi(4) * dot_k_hat_omega_hat.powi(2);
    
    // Suppression of small waves (capillary waves)
    let l_squared = 0.05; // small ripples cutoff factor - increased to reduce high frequency components
    result *= (-k_length * k_length * l_squared).exp();
    
    result
}

/// Generate initial wave spectrum with complex amplitudes
fn generate_h0(params: &OceanParams) -> Vec<Vec<Complex32>> {
    let n = params.resolution;
    let mut h0 = vec![vec![Complex32::new(0.0, 0.0); n]; n];
    let mut rng = StdRng::seed_from_u64(params.seed);
    
    for i in 0..n {
        for j in 0..n {
            // Calculate wave vector k using FFT frequency ordering
            let k_x = 2.0 * PI * fft_freq(i, n) / params.size;
            let k_y = 2.0 * PI * fft_freq(j, n) / params.size;
            let k = Vec2::new(k_x, k_y);

            // Calculate Phillips spectrum
            let phillips = phillips_spectrum(k, params);
            
            // Generate complex Gaussian random variables using normal distribution
            // Standard normal distribution can be approximated with two random values
            let xi_r = (rng.gen::<f32>() * 2.0 - 1.0) + (rng.gen::<f32>() * 2.0 - 1.0);
            let xi_i = (rng.gen::<f32>() * 2.0 - 1.0) + (rng.gen::<f32>() * 2.0 - 1.0);
            
            // Combine into complex amplitude (Eq. 25 in Tessendorf paper)
            h0[i][j] = Complex32::new(xi_r, xi_i) * 0.707 * phillips.sqrt();
        }
    }
    
    h0
}

/// Calculate dispersion relation for deep water waves
fn dispersion(k: f32, params: &OceanParams) -> f32 {
    // Dispersion relation for infinite depth
    let base_dispersion = (params.gravity * k).sqrt();
    
    // For finite depth (if depth parameter is set)
    if params.depth < 100.0 {
        // Modified dispersion relation for finite depth
        let tanh_term = (k * params.depth).tanh();
        return (params.gravity * k * tanh_term).sqrt();
    }
    
    base_dispersion
}

/// Unscaled spatial-domain fields for one instant.
struct RawFields {
    height: Vec<Vec<f32>>,
    slope_x: Vec<Vec<f32>>,
    slope_y: Vec<Vec<f32>>,
    disp_x: Vec<Vec<f32>>,
    disp_y: Vec<Vec<f32>>,
}

/// Evolve `h0` to `time` and inverse-transform all fields.
fn raw_fields(
    params: &OceanParams,
    h0: &[Vec<Complex32>],
    time: f32,
    planner: &mut FftPlanner<f32>,
) -> RawFields {
    let n = params.resolution;
    let zero = || vec![vec![Complex32::new(0.0, 0.0); n]; n];
    let mut height_field = zero();
    let mut slope_x_field = zero();
    let mut slope_y_field = zero();
    let mut disp_x_field = zero();
    let mut disp_y_field = zero();

    for i in 0..n {
        for j in 0..n {
            let k_x = 2.0 * PI * fft_freq(i, n) / params.size;
            let k_y = 2.0 * PI * fft_freq(j, n) / params.size;
            let k = Vec2::new(k_x, k_y);
            let k_length = k.length();
            let k_hat = if k_length > 1e-6 { k / k_length } else { Vec2::new(1.0, 0.0) };

            let omega = dispersion(k_length, params);
            let exp_term = Complex32::new(0.0, omega * time).exp();
            let exp_term_conj = Complex32::new(0.0, -omega * time).exp();

            let h0_k = h0[i][j];
            let h0_conj_neg_k = Complex32::new(h0_k.re, -h0_k.im);
            let h_tilde = h0_k * exp_term + h0_conj_neg_k * exp_term_conj;

            height_field[i][j] = h_tilde; // Eq. 19
            slope_x_field[i][j] = Complex32::new(0.0, k.x) * h_tilde; // Eq. 20
            slope_y_field[i][j] = Complex32::new(0.0, k.y) * h_tilde;
            disp_x_field[i][j] = Complex32::new(0.0, -k_hat.x) * h_tilde * params.choppiness; // Eq. 29
            disp_y_field[i][j] = Complex32::new(0.0, -k_hat.y) * h_tilde * params.choppiness;
        }
    }

    RawFields {
        height: ifft2d(planner, height_field),
        slope_x: ifft2d(planner, slope_x_field),
        slope_y: ifft2d(planner, slope_y_field),
        disp_x: ifft2d(planner, disp_x_field),
        disp_y: ifft2d(planner, disp_y_field),
    }
}

fn height_range(field: &[Vec<f32>]) -> f32 {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for v in field.iter().flatten() {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    hi - lo
}

fn scale_all(field: &mut [Vec<f32>], scale: f32) {
    field.iter_mut().flatten().for_each(|v| *v *= scale);
}

/// Apply one amplitude `scale` to every field (scaling the spectrum is linear, so
/// height, slope and displacement stay mutually consistent) and derive normals.
fn finish_field(mut raw: RawFields, scale: f32) -> DisplacementField {
    let n = raw.height.len();
    for f in [&mut raw.height, &mut raw.slope_x, &mut raw.slope_y, &mut raw.disp_x, &mut raw.disp_y] {
        scale_all(f, scale);
    }

    let mut normal_x = vec![vec![0.0; n]; n];
    let mut normal_y = vec![vec![0.0; n]; n];
    let mut normal_z = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let (nx, ny, nz) = (-raw.slope_x[i][j], -raw.slope_y[i][j], 1.0f32);
            let length = (nx * nx + ny * ny + nz * nz).sqrt();
            normal_x[i][j] = nx / length;
            normal_y[i][j] = ny / length;
            normal_z[i][j] = nz / length;
        }
    }

    DisplacementField {
        resolution: n,
        height: raw.height,
        displacement_x: raw.disp_x,
        displacement_y: raw.disp_y,
        normal_x,
        normal_y,
        normal_z,
    }
}

/// Keyframe times (seconds of simulation time) for an animation covering
/// `[start, start + duration]` sampled at `fps`.
///
/// Both endpoints are included, so the result has `round(duration * fps) + 1` entries.
pub fn frame_times(start: f32, duration: f32, fps: f32) -> Result<Vec<f32>> {
    ensure!(start.is_finite(), "animation start must be finite, got {start}");
    ensure!(duration.is_finite() && duration > 0.0, "animation duration must be > 0, got {duration}");
    ensure!(fps.is_finite() && fps > 0.0, "animation fps must be > 0, got {fps}");
    let intervals = (duration * fps).round() as usize;
    ensure!(
        intervals >= 1,
        "duration {duration}s at {fps} fps yields no full frame interval; increase duration or fps"
    );
    Ok((0..=intervals).map(|i| start + i as f32 / fps).collect())
}

/// Compute displacement fields at several instants of the same ocean.
///
/// The spectrum is generated once and one amplitude scale is shared by all frames,
/// chosen so the *largest* peak-to-peak height over all frames is `2 * amplitude`.
/// Frames are therefore mutually consistent (heights do not pump between frames).
///
/// # Errors
/// If `times` is empty or non-finite, or `params` has a resolution below 2 or a
/// non-positive size.
pub fn compute_wave_fields(params: &OceanParams, times: &[f32]) -> Result<Vec<DisplacementField>> {
    ensure!(!times.is_empty(), "at least one time is required");
    ensure!(times.iter().all(|t| t.is_finite()), "times must be finite");
    ensure!(params.resolution >= 2, "resolution must be at least 2, got {}", params.resolution);
    ensure!(params.size.is_finite() && params.size > 0.0, "size must be > 0, got {}", params.size);

    let h0 = generate_h0(params);
    let mut planner = FftPlanner::new();
    let raws: Vec<RawFields> = times.iter().map(|&t| raw_fields(params, &h0, t, &mut planner)).collect();

    let current_range = raws.iter().map(|r| height_range(&r.height)).fold(0.0f32, f32::max);
    let target_range = params.amplitude * 2.0; // peak-to-peak should be 2 * amplitude
    let scale = if current_range > 1e-6 {
        target_range / current_range
    } else {
        // No variation at all; keep the historical fallback.
        params.amplitude
    };
    println!(
        "Wave height scaling: {} frame(s), raw range = {:.6}, target = {:.6}, scale = {:.6}",
        raws.len(), current_range, target_range, scale
    );

    Ok(raws.into_iter().map(|r| finish_field(r, scale)).collect())
}

/// Compute the wave field at a single time (see [`compute_wave_fields`]).
///
/// # Panics
/// If `params` is invalid (see [`compute_wave_fields`]) or `time` is not finite.
pub fn compute_wave_field(params: &OceanParams, time: f32) -> DisplacementField {
    compute_wave_fields(params, &[time])
        .expect("invalid OceanParams or time")
        .remove(0)
}

/// Perform 2D inverse FFT on complex field via row-then-column 1D IFFTs.
/// rustfft is un-normalized, so we divide by n*n at the end (one factor
/// of n per dimension).
fn ifft2d(planner: &mut FftPlanner<f32>, mut field: Vec<Vec<Complex32>>) -> Vec<Vec<f32>> {
    let n = field.len();
    let fft = planner.plan_fft_inverse(n);

    // IFFT along rows (j axis)
    for row in field.iter_mut() {
        fft.process(row);
    }

    // IFFT along columns (i axis): transpose, process, transpose back
    let mut transposed = vec![vec![Complex32::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            transposed[j][i] = field[i][j];
        }
    }
    for col in transposed.iter_mut() {
        fft.process(col);
    }

    let scale = 1.0 / (n * n) as f32;
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = transposed[j][i].re * scale;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field() -> DisplacementField {
        let params = OceanParams { resolution: 32, size: 20.0, ..OceanParams::default() };
        compute_wave_field(&params, 1.3)
    }

    #[test]
    fn sample_wraps_exactly_at_period() {
        let f = field();
        let n = f.resolution as f64;
        for k in [0.0, 3.0, 7.5, 31.9] {
            assert_eq!(f.sample_periodic(0.0, k), f.sample_periodic(n, k));
            assert_eq!(f.sample_periodic(k, 0.0), f.sample_periodic(k, n));
            assert_eq!(f.sample_periodic(k, k), f.sample_periodic(k + n, k - n));
        }
    }

    fn range(f: &DisplacementField) -> f32 {
        height_range(&f.height)
    }

    #[test]
    fn single_frame_hits_commanded_peak_to_peak() {
        let params = OceanParams { resolution: 32, size: 20.0, amplitude: 1.5, ..OceanParams::default() };
        let f = compute_wave_field(&params, 0.4);
        assert!((range(&f) - 3.0).abs() < 1e-3, "range {}", range(&f));
    }

    #[test]
    fn sequence_shares_one_scale() {
        let params = OceanParams { resolution: 32, size: 20.0, amplitude: 1.0, ..OceanParams::default() };
        let times = frame_times(0.0, 2.0, 10.0).unwrap();
        let frames = compute_wave_fields(&params, &times).unwrap();
        let ranges: Vec<f32> = frames.iter().map(range).collect();
        let max = ranges.iter().cloned().fold(0.0, f32::max);
        assert!((max - 2.0).abs() < 1e-3, "max range {max}");
        // Frames are NOT individually normalised.
        assert!(ranges.iter().any(|r| (r - 2.0).abs() > 1e-3), "{ranges:?}");
        // A frame inside a sequence equals the same instant in a one-element sequence only
        // up to the shared scale: ratios of heights between frames must match the raw ratios.
        let single = compute_wave_fields(&params, &times[3..4]).unwrap().remove(0);
        let ratio = frames[3].height[2][5] / single.height[2][5];
        let ratio2 = frames[3].height[7][1] / single.height[7][1];
        assert!((ratio - ratio2).abs() < 1e-3 * ratio.abs().max(1.0));
    }

    #[test]
    fn normals_follow_scaled_slopes() {
        // Steeper (larger amplitude) ocean must tilt normals more.
        let mk = |a| {
            let p = OceanParams { resolution: 32, size: 20.0, amplitude: a, ..OceanParams::default() };
            compute_wave_field(&p, 0.0)
        };
        let tilt = |f: &DisplacementField| f.normal_z.iter().flatten().cloned().fold(1.0f32, f32::min);
        assert!(tilt(&mk(2.0)) < tilt(&mk(0.2)));
    }

    #[test]
    fn frame_times_include_both_endpoints() {
        let t = frame_times(1.0, 2.0, 4.0).unwrap();
        assert_eq!(t.len(), 9);
        assert_eq!(t[0], 1.0);
        assert_eq!(*t.last().unwrap(), 3.0);
    }

    #[test]
    fn frame_times_reject_bad_input() {
        assert!(frame_times(0.0, 0.0, 24.0).is_err());
        assert!(frame_times(0.0, -1.0, 24.0).is_err());
        assert!(frame_times(0.0, 1.0, 0.0).is_err());
        assert!(frame_times(0.0, 1.0, f32::NAN).is_err());
        assert!(frame_times(f32::INFINITY, 1.0, 24.0).is_err());
        assert!(frame_times(0.0, 0.01, 10.0).is_err()); // rounds to zero intervals
    }

    #[test]
    fn compute_rejects_bad_params() {
        let bad = OceanParams { resolution: 1, ..OceanParams::default() };
        assert!(compute_wave_fields(&bad, &[0.0]).is_err());
        assert!(compute_wave_fields(&OceanParams::default(), &[]).is_err());
        assert!(compute_wave_fields(&OceanParams::default(), &[f32::NAN]).is_err());
    }

    #[test]
    fn sample_on_grid_point_matches_field() {
        let f = field();
        let s = f.sample_periodic(5.0, 9.0);
        assert_eq!(s.height, f.height[5][9]);
        assert_eq!(s.displacement, Vec2::new(f.displacement_x[5][9], f.displacement_y[5][9]));
    }

    #[test]
    fn interpolation_stays_within_neighbours() {
        let f = field();
        let s = f.sample_periodic(5.5, 9.0);
        let (a, b) = (f.height[5][9], f.height[6][9]);
        assert!(s.height >= a.min(b) - 1e-6 && s.height <= a.max(b) + 1e-6);
    }
}
