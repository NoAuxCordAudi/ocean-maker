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

/// Sign of the choppy horizontal displacement spectrum: `D~ = SIGN * i * k_hat * h~`.
///
/// With rustfft's `e^{+ik.x}` inverse transform, `+i` makes `D = -k_hat * a * sin(theta)` for
/// `h = a * cos(theta)`, i.e. horizontal motion converges on crests (they sharpen) as in real
/// deep-water waves. The Jacobian uses the same constant so mesh and foam always agree. See
/// `docs/whitecaps-spikes.md`; [`verify_fold_sign`] checks it on real output.
const DISPLACEMENT_SIGN: f32 = 1.0;

/// Unscaled spatial-domain fields for one instant.
struct RawFields {
    height: Vec<Vec<f32>>,
    slope_x: Vec<Vec<f32>>,
    slope_y: Vec<Vec<f32>>,
    disp_x: Vec<Vec<f32>>,
    disp_y: Vec<Vec<f32>>,
}

/// Wave vector of FFT bin `(i, j)`, its unit direction, and the evolved spectrum `h~(k, t)`.
fn spectrum_at(
    params: &OceanParams,
    h0: &[Vec<Complex32>],
    i: usize,
    j: usize,
    time: f32,
) -> (Vec2, Vec2, Complex32) {
    let n = params.resolution;
    let k = Vec2::new(
        2.0 * PI * fft_freq(i, n) / params.size,
        2.0 * PI * fft_freq(j, n) / params.size,
    );
    let k_length = k.length();
    let k_hat = if k_length > 1e-6 { k / k_length } else { Vec2::new(1.0, 0.0) };

    let omega = dispersion(k_length, params);
    let exp_term = Complex32::new(0.0, omega * time).exp();
    let exp_term_conj = Complex32::new(0.0, -omega * time).exp();

    let h0_k = h0[i][j];
    let h0_conj_neg_k = Complex32::new(h0_k.re, -h0_k.im);
    (k, k_hat, h0_k * exp_term + h0_conj_neg_k * exp_term_conj)
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
            let (k, k_hat, h_tilde) = spectrum_at(params, h0, i, j, time);
            height_field[i][j] = h_tilde; // Eq. 19
            slope_x_field[i][j] = Complex32::new(0.0, k.x) * h_tilde; // Eq. 20
            slope_y_field[i][j] = Complex32::new(0.0, k.y) * h_tilde;
            // Eq. 29, sign per DISPLACEMENT_SIGN
            disp_x_field[i][j] =
                Complex32::new(0.0, DISPLACEMENT_SIGN * k_hat.x) * h_tilde * params.choppiness;
            disp_y_field[i][j] =
                Complex32::new(0.0, DISPLACEMENT_SIGN * k_hat.y) * h_tilde * params.choppiness;
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

fn ensure_finite_times(times: &[f32]) -> Result<()> {
    ensure!(!times.is_empty(), "at least one time is required");
    ensure!(times.iter().all(|t| t.is_finite()), "times must be finite");
    Ok(())
}

/// Jacobian of the horizontal map `x -> x + D(x)` on a (possibly oversampled) grid.
#[derive(Debug, Clone)]
pub struct JacobianField {
    pub resolution: usize,
    /// Smaller eigenvalue of the symmetric Jacobian `[[1+Dxx, Dxy], [Dxy, 1+Dyy]]`.
    /// 1 = undistorted, 0 = crest of a fold forming, < 0 = the surface has folded over.
    pub lambda_min: Vec<Vec<f32>>,
    /// `det J`; negative where the displaced mesh folds (a diagnostic).
    pub det: Vec<Vec<f32>>,
}

/// One ocean realisation (spectrum fixed by `params.seed`) that can be evaluated at any time.
///
/// All evaluations must share one height `scale` (see [`OceanSim::height_scale`]) so that
/// frames, and fields derived from them, are mutually consistent.
pub struct OceanSim {
    params: OceanParams,
    h0: Vec<Vec<Complex32>>,
}

impl OceanSim {
    /// # Errors
    /// If `params` has a resolution below 2 or a non-positive size.
    pub fn new(params: &OceanParams) -> Result<Self> {
        Self::validate(params)?;
        Ok(Self { params: params.clone(), h0: generate_h0(params) })
    }

    fn validate(params: &OceanParams) -> Result<()> {
        ensure!(params.resolution >= 2, "resolution must be at least 2, got {}", params.resolution);
        ensure!(params.size.is_finite() && params.size > 0.0, "size must be > 0, got {}", params.size);
        Ok(())
    }

    #[cfg(test)]
    fn with_h0(params: &OceanParams, h0: Vec<Vec<Complex32>>) -> Self {
        Self { params: params.clone(), h0 }
    }

    /// Simulation grid resolution.
    pub fn resolution(&self) -> usize {
        self.params.resolution
    }

    /// Choppiness factor.
    pub fn choppiness(&self) -> f32 {
        self.params.choppiness
    }

    /// The evolved spectrum `h~(k, t)` in FFT bin order (`[i][j]`, bin `i` = x, `j` = y), before
    /// any scaling. Exposed so independent implementations can be checked against this one.
    pub fn spectrum(&self, time: f32) -> Vec<Vec<Complex32>> {
        let n = self.params.resolution;
        (0..n)
            .map(|i| (0..n).map(|j| spectrum_at(&self.params, &self.h0, i, j, time).2).collect())
            .collect()
    }

    /// Single height scale such that the largest peak-to-peak height over `times` is
    /// `2 * amplitude`.
    ///
    /// # Errors
    /// If `times` is empty or non-finite.
    pub fn height_scale(&self, times: &[f32]) -> Result<f32> {
        ensure_finite_times(times)?;
        let mut planner = FftPlanner::new();
        let n = self.params.resolution;
        let mut current_range = 0.0f32;
        for &t in times {
            let mut field = vec![vec![Complex32::new(0.0, 0.0); n]; n];
            for (i, row) in field.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell = spectrum_at(&self.params, &self.h0, i, j, t).2;
                }
            }
            current_range = current_range.max(height_range(&ifft2d(&mut planner, field)));
        }
        let target_range = self.params.amplitude * 2.0;
        Ok(if current_range > 1e-6 { target_range / current_range } else { self.params.amplitude })
    }

    /// Displacement fields at `times`, all scaled by `scale`.
    ///
    /// # Errors
    /// If `times` is empty or non-finite.
    pub fn fields(&self, times: &[f32], scale: f32) -> Result<Vec<DisplacementField>> {
        ensure_finite_times(times)?;
        let mut planner = FftPlanner::new();
        Ok(times
            .iter()
            .map(|&t| finish_field(raw_fields(&self.params, &self.h0, t, &mut planner), scale))
            .collect())
    }

    /// Jacobian of the displacement map at `times`, on an `out_resolution`² grid.
    ///
    /// The three second-derivative fields `D_ab = IFFT[-SIGN * c * s * k_a k_b / |k| * h~]` are
    /// computed spectrally (exact and periodic). When `out_resolution > resolution` the spectrum
    /// is zero-padded, which band-limit interpolates the *linear* fields; the nonlinear
    /// eigenvalue is then evaluated on the finer grid. Values at coincident grid points equal the
    /// coarse values (the unnormalised sum is unchanged, so no `(M/N)^2` factor is needed).
    /// Output texel `(i, j)` lies at `x = (i + sample_offset) / out_resolution * size`; use
    /// `sample_offset = 0.5` for texel *centres* (image textures), `0.0` for grid points. The
    /// shift is applied exactly in the spectrum.
    ///
    /// The Nyquist row/column is dropped (its derivative is ill-defined).
    ///
    /// # Errors
    /// If `times` is empty/non-finite or `out_resolution < resolution`.
    pub fn jacobians(
        &self,
        times: &[f32],
        scale: f32,
        out_resolution: usize,
        sample_offset: f32,
    ) -> Result<Vec<JacobianField>> {
        ensure_finite_times(times)?;
        ensure!(sample_offset.is_finite(), "sample_offset must be finite");
        let n = self.params.resolution;
        ensure!(out_resolution >= n, "out_resolution {out_resolution} must be >= resolution {n}");
        let mut planner = FftPlanner::new();
        let c = self.params.choppiness * scale;
        // Spatial shift of `delta` world units = multiply bin k by e^{+i k . delta}.
        let delta = sample_offset * self.params.size / out_resolution as f32;
        let is_nyquist = |idx: usize| n.is_multiple_of(2) && idx == n / 2;

        Ok(times
            .iter()
            .map(|&t| {
                let zero = || vec![vec![Complex32::new(0.0, 0.0); n]; n];
                let (mut xx, mut yy, mut xy) = (zero(), zero(), zero());
                for i in 0..n {
                    for j in 0..n {
                        if is_nyquist(i) || is_nyquist(j) {
                            continue;
                        }
                        let (k, _, h) = spectrum_at(&self.params, &self.h0, i, j, t);
                        let kl = k.length();
                        if kl < 1e-6 {
                            continue;
                        }
                        let m = -DISPLACEMENT_SIGN * c / kl;
                        let h = h * Complex32::new(0.0, (k.x + k.y) * delta).exp();
                        xx[i][j] = h * (m * k.x * k.x);
                        yy[i][j] = h * (m * k.y * k.y);
                        xy[i][j] = h * (m * k.x * k.y);
                    }
                }
                // Keep the coarse 1/N^2 normalisation so values match the coarse fields.
                let norm = 1.0 / (n * n) as f32;
                let dxx = ifft_square(&mut planner, pad_spectrum(&xx, out_resolution), norm);
                let dyy = ifft_square(&mut planner, pad_spectrum(&yy, out_resolution), norm);
                let dxy = ifft_square(&mut planner, pad_spectrum(&xy, out_resolution), norm);

                let m = out_resolution;
                let mut lambda_min = vec![vec![0.0f32; m]; m];
                let mut det = vec![vec![0.0f32; m]; m];
                for i in 0..m {
                    for j in 0..m {
                        let (jxx, jyy, jxy) = (1.0 + dxx[i][j] as f64, 1.0 + dyy[i][j] as f64, dxy[i][j] as f64);
                        let disc = ((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy).sqrt();
                        lambda_min[i][j] = (0.5 * (jxx + jyy) - 0.5 * disc) as f32;
                        det[i][j] = (jxx * jyy - jxy * jxy) as f32;
                    }
                }
                JacobianField { resolution: m, lambda_min, det }
            })
            .collect())
    }
}

/// Embed an `n`x`n` FFT-ordered spectrum into an `m`x`m` one (m >= n): positive frequencies stay,
/// negative frequencies move to the top of the larger array. The Nyquist bin (if any) must
/// already be zero.
fn pad_spectrum(src: &[Vec<Complex32>], m: usize) -> Vec<Vec<Complex32>> {
    let n = src.len();
    if m == n {
        return src.to_vec();
    }
    let map = |i: usize| if i < n / 2 + n % 2 { i } else { m - (n - i) };
    let mut out = vec![vec![Complex32::new(0.0, 0.0); m]; m];
    for (i, row) in src.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[map(i)][map(j)] = *v;
        }
    }
    out
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
    ensure_finite_times(times)?;
    let sim = OceanSim::new(params)?;
    let scale = sim.height_scale(times)?;
    println!(
        "Wave height scaling: {} frame(s), target = {:.6}, scale = {:.6}",
        times.len(),
        params.amplitude * 2.0,
        scale
    );
    sim.fields(times, scale)
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

/// Check that the choppy displacement sharpens crests: the correlation between height and the
/// horizontal divergence of `D` over the grid must be negative (crests compress). Returns the
/// correlation. Uses periodic central differences, so it is a coarse but independent check of
/// [`DISPLACEMENT_SIGN`] and of the Jacobian's sign convention.
///
/// # Errors
/// If the correlation is not clearly negative (the sign convention is broken).
pub fn verify_fold_sign(params: &OceanParams, time: f32) -> Result<f32> {
    ensure!(params.choppiness > 0.0, "choppiness must be > 0 to test the displacement sign");
    let f = compute_wave_field(params, time);
    let n = f.resolution;
    let dx = params.size / n as f32;
    let w = |i: isize| (i.rem_euclid(n as isize)) as usize;
    let (mut hs, mut ds) = (Vec::with_capacity(n * n), Vec::with_capacity(n * n));
    for i in 0..n as isize {
        for j in 0..n as isize {
            let ddx = (f.displacement_x[w(i + 1)][w(j)] - f.displacement_x[w(i - 1)][w(j)]) / (2.0 * dx);
            let ddy = (f.displacement_y[w(i)][w(j + 1)] - f.displacement_y[w(i)][w(j - 1)]) / (2.0 * dx);
            hs.push(f.height[i as usize][j as usize] as f64);
            ds.push((ddx + ddy) as f64);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mh, md) = (mean(&hs), mean(&ds));
    let cov: f64 = hs.iter().zip(&ds).map(|(h, d)| (h - mh) * (d - md)).sum();
    let vh: f64 = hs.iter().map(|h| (h - mh).powi(2)).sum();
    let vd: f64 = ds.iter().map(|d| (d - md).powi(2)).sum();
    let corr = (cov / (vh.sqrt() * vd.sqrt())) as f32;
    ensure!(
        corr < -0.1,
        "choppy displacement does not sharpen crests (corr(height, div D) = {corr:+.3}); sign convention is wrong"
    );
    Ok(corr)
}

/// Perform a 2D inverse FFT of the coarse `n`x`n` field, normalised by `1/n^2`.
fn ifft2d(planner: &mut FftPlanner<f32>, field: Vec<Vec<Complex32>>) -> Vec<Vec<f32>> {
    let n = field.len();
    ifft_square(planner, field, 1.0 / (n * n) as f32)
}

/// 2D inverse FFT of a square field via row-then-column 1D IFFTs, returning the real part times
/// `scale`. rustfft is un-normalised, so the caller chooses the normalisation.
fn ifft_square(planner: &mut FftPlanner<f32>, mut field: Vec<Vec<Complex32>>, scale: f32) -> Vec<Vec<f32>> {
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

    // ---- Jacobian / sign (phase 1) ----

    fn mode_params(c: f32) -> OceanParams {
        OceanParams { resolution: 64, size: 20.0, choppiness: c, amplitude: 1.0, ..OceanParams::default() }
    }

    /// h0 with the given (i, j, value) bins set, everything else zero.
    fn h0_with(n: usize, bins: &[(usize, usize, Complex32)]) -> Vec<Vec<Complex32>> {
        let mut h0 = vec![vec![Complex32::new(0.0, 0.0); n]; n];
        for &(i, j, v) in bins {
            h0[i][j] = v;
        }
        h0
    }

    #[test]
    fn crests_compress_on_real_output() {
        let params = OceanParams { resolution: 64, size: 20.0, choppiness: 0.5, ..OceanParams::default() };
        let corr = verify_fold_sign(&params, 0.0).unwrap();
        assert!(corr < -0.1, "corr {corr}");
    }

    #[test]
    fn single_mode_matches_analytic_jacobian() {
        let (n, c) = (64usize, 0.8f32);
        let params = mode_params(c);
        // One x-directed mode, bin (3, 0): h = A cos(kx), A = 2a / N^2 at t = 0.
        let a = 40.0f32;
        let sim = OceanSim::with_h0(&params, h0_with(n, &[(3, 0, Complex32::new(a, 0.0))]));
        let amp = 2.0 * a / (n * n) as f32;
        let k = 2.0 * PI * 3.0 / params.size;

        let field = sim.fields(&[0.0], 1.0).unwrap().remove(0);
        let jac = sim.jacobians(&[0.0], 1.0, n, 0.0).unwrap().remove(0);
        for i in 0..n {
            let x = i as f32 / n as f32 * params.size;
            let theta = k * x;
            // crest at theta = 0: height max, displacement compresses (J < 1)
            assert!((field.height[i][5] - amp * theta.cos()).abs() < 1e-6);
            assert!((field.displacement_x[i][5] + c * amp * theta.sin()).abs() < 1e-6);
            let want_det = 1.0 - c * amp * k * theta.cos();
            assert!((jac.det[i][5] - want_det).abs() < 1e-5, "i={i} {} vs {want_det}", jac.det[i][5]);
            assert!((jac.lambda_min[i][5] - want_det.min(1.0)).abs() < 1e-5);
        }
        // crest sharpening: smallest J exactly at the crest
        assert!(jac.det[0][5] < 1.0 && jac.det[n / 6][5] > 1.0); // theta = 0 vs pi
    }

    #[test]
    fn zero_choppiness_gives_identity_jacobian() {
        let params = mode_params(0.0);
        let sim = OceanSim::new(&params).unwrap();
        let j = sim.jacobians(&[0.3], 5.0, 64, 0.0).unwrap().remove(0);
        assert!(j.lambda_min.iter().flatten().all(|&v| v == 1.0));
        assert!(j.det.iter().flatten().all(|&v| v == 1.0));
    }

    #[test]
    fn spectral_jacobian_matches_finite_differences_on_low_k_field() {
        let (n, c) = (64usize, 1.0f32);
        let params = mode_params(c);
        let bins = [
            (2, 1, Complex32::new(30.0, 10.0)),
            (1, 3, Complex32::new(-20.0, 25.0)),
            (5, 2, Complex32::new(15.0, -5.0)),
            (3, 62, Complex32::new(10.0, 20.0)), // negative ky
            (60, 4, Complex32::new(-12.0, 8.0)), // negative kx
        ];
        let sim = OceanSim::with_h0(&params, h0_with(n, &bins));
        let scale = 25.0;
        let field = sim.fields(&[0.7], scale).unwrap().remove(0);
        let jac = sim.jacobians(&[0.7], scale, n, 0.0).unwrap().remove(0);

        let dx = params.size / n as f32;
        let w = |i: isize| i.rem_euclid(n as isize) as usize;
        let mut max_err = 0.0f32;
        let mut max_dev = 0.0f32;
        let mut max_dj = 0.0f32; // largest |det J - 1|, to scale the FD tolerance
        for i in 0..n as isize {
            for j in 0..n as isize {
                let d = |f: &Vec<Vec<f32>>, di: isize, dj: isize| f[w(i + di)][w(j + dj)];
                let jxx = 1.0 + (d(&field.displacement_x, 1, 0) - d(&field.displacement_x, -1, 0)) / (2.0 * dx);
                let jyy = 1.0 + (d(&field.displacement_y, 0, 1) - d(&field.displacement_y, 0, -1)) / (2.0 * dx);
                let jxy = (d(&field.displacement_x, 0, 1) - d(&field.displacement_x, 0, -1)) / (2.0 * dx);
                let jyx = (d(&field.displacement_y, 1, 0) - d(&field.displacement_y, -1, 0)) / (2.0 * dx);
                // D is a gradient field, so the Jacobian is symmetric.
                max_dev = max_dev.max((jxy - jyx).abs());
                let det_fd = jxx * jyy - jxy * jyx;
                max_err = max_err.max((det_fd - jac.det[i as usize][j as usize]).abs());
                max_dj = max_dj.max((jac.det[i as usize][j as usize] - 1.0).abs());
            }
        }
        // Central differences attenuate wave number k by sin(k dx)/(k dx) (0.96 at the highest
        // mode here), so allow 8% of the signal; a wrong sign or factor would be ~100%.
        assert!(max_dj > 0.2, "test field too weak: {max_dj}");
        assert!(max_dev < 0.08 * max_dj, "Dxy != Dyx by {max_dev}");
        assert!(max_err < 0.08 * max_dj, "spectral vs FD det differ by {max_err} (signal {max_dj})");
    }

    #[test]
    fn oversampled_jacobian_matches_coarse_at_coincident_samples() {
        let params = OceanParams { resolution: 64, size: 20.0, choppiness: 1.5, ..OceanParams::default() };
        let sim = OceanSim::new(&params).unwrap();
        let scale = sim.height_scale(&[0.4]).unwrap();
        let coarse = sim.jacobians(&[0.4], scale, 64, 0.0).unwrap().remove(0);
        let fine = sim.jacobians(&[0.4], scale, 128, 0.0).unwrap().remove(0);
        assert_eq!(fine.resolution, 128);
        let mut max_err = 0.0f32;
        for i in 0..64 {
            for j in 0..64 {
                max_err = max_err.max((fine.det[2 * i][2 * j] - coarse.det[i][j]).abs());
                max_err = max_err.max((fine.lambda_min[2 * i][2 * j] - coarse.lambda_min[i][j]).abs());
            }
        }
        // Not 4x inflated (that would be O(1)): equal up to f32 round-off.
        assert!(max_err < 1e-4, "{max_err}");
    }

    #[test]
    fn oversampling_resolves_finer_structure_than_coarse() {
        let params = OceanParams { resolution: 32, size: 20.0, choppiness: 2.0, ..OceanParams::default() };
        let sim = OceanSim::new(&params).unwrap();
        let scale = sim.height_scale(&[0.0]).unwrap();
        let coarse = sim.jacobians(&[0.0], scale, 32, 0.0).unwrap().remove(0);
        let fine = sim.jacobians(&[0.0], scale, 128, 0.0).unwrap().remove(0);
        let min = |j: &JacobianField| j.lambda_min.iter().flatten().cloned().fold(f32::MAX, f32::min);
        // Between-sample peaks can only make the minimum lower or equal.
        assert!(min(&fine) <= min(&coarse) + 1e-6);
    }

    #[test]
    fn folded_mesh_has_negative_lambda_min() {
        let params = OceanParams { resolution: 64, size: 20.0, choppiness: 2.5, ..OceanParams::default() };
        let sim = OceanSim::new(&params).unwrap();
        let scale = sim.height_scale(&[0.0]).unwrap();
        let j = sim.jacobians(&[0.0], scale, 64, 0.0).unwrap().remove(0);
        let folded = j.det.iter().flatten().filter(|&&d| d < 0.0).count();
        assert!(folded > 0, "expected folds at high choppiness");
        for (d, l) in j.det.iter().flatten().zip(j.lambda_min.iter().flatten()) {
            if *d < 0.0 {
                assert!(*l < 0.0);
            }
        }
    }

    #[test]
    fn sample_offset_shifts_texel_centres() {
        let params = OceanParams { resolution: 64, size: 20.0, choppiness: 1.5, ..OceanParams::default() };
        let sim = OceanSim::new(&params).unwrap();
        let scale = sim.height_scale(&[0.4]).unwrap();
        // Half-texel-offset 128 grid == odd samples of the plain 256 grid.
        let shifted = sim.jacobians(&[0.4], scale, 128, 0.5).unwrap().remove(0);
        let fine = sim.jacobians(&[0.4], scale, 256, 0.0).unwrap().remove(0);
        let mut max_err = 0.0f32;
        for i in 0..128 {
            for j in 0..128 {
                max_err = max_err.max((shifted.det[i][j] - fine.det[2 * i + 1][2 * j + 1]).abs());
            }
        }
        assert!(max_err < 1e-4, "{max_err}");
        // and it is genuinely different from the unshifted grid
        let plain = sim.jacobians(&[0.4], scale, 128, 0.0).unwrap().remove(0);
        assert!(plain.det != shifted.det);
    }

    #[test]
    fn jacobians_reject_bad_input() {
        let sim = OceanSim::new(&mode_params(0.5)).unwrap();
        assert!(sim.jacobians(&[], 1.0, 64, 0.0).is_err());
        assert!(sim.jacobians(&[0.0], 1.0, 32, 0.0).is_err()); // below native resolution
        assert!(sim.jacobians(&[f32::NAN], 1.0, 64, 0.0).is_err());
    }

    #[test]
    fn sim_is_deterministic() {
        let a = OceanSim::new(&mode_params(1.0)).unwrap().jacobians(&[1.0], 3.0, 64, 0.0).unwrap();
        let b = OceanSim::new(&mode_params(1.0)).unwrap().jacobians(&[1.0], 3.0, 64, 0.0).unwrap();
        assert_eq!(a[0].lambda_min, b[0].lambda_min);
    }
}
