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

/// Compute wave height field at a given time
pub fn compute_wave_field(params: &OceanParams, time: f32) -> DisplacementField {
    let n = params.resolution;
    
    // Generate initial spectrum h0
    let h0 = generate_h0(params);
    
    // Initialize FFT input fields
    let mut height_field = vec![vec![Complex32::new(0.0, 0.0); n]; n];
    let mut slope_x_field = vec![vec![Complex32::new(0.0, 0.0); n]; n];
    let mut slope_y_field = vec![vec![Complex32::new(0.0, 0.0); n]; n];
    let mut disp_x_field = vec![vec![Complex32::new(0.0, 0.0); n]; n];
    let mut disp_y_field = vec![vec![Complex32::new(0.0, 0.0); n]; n];
    
    // Compute height field and derivatives for the given time
    for i in 0..n {
        for j in 0..n {
            // Calculate wave vector k using FFT frequency ordering
            let k_x = 2.0 * PI * fft_freq(i, n) / params.size;
            let k_y = 2.0 * PI * fft_freq(j, n) / params.size;
            let k = Vec2::new(k_x, k_y);
            let k_length = k.length();
            
            // Normalize k vector (avoiding division by zero)
            let k_hat = if k_length > 1e-6 {
                k / k_length
            } else {
                Vec2::new(1.0, 0.0)
            };
            
            // Calculate dispersion
            let omega = dispersion(k_length, params);
            
            // Complex exponential for time evolution
            let exp_term = Complex32::new(0.0, omega * time).exp();
            let exp_term_conj = Complex32::new(0.0, -omega * time).exp();
            
            // Calculate h_tilde(k, t) and its complex conjugate
            let h0_k = h0[i][j];
            let h0_conj_neg_k = Complex32::new(h0[i][j].re, -h0[i][j].im);
            
            let h_tilde = h0_k * exp_term + h0_conj_neg_k * exp_term_conj;
            
            // Height field (Eq. 19 in Tessendorf paper)
            height_field[i][j] = h_tilde;
            
            // Slope field (Eq. 20)
            slope_x_field[i][j] = Complex32::new(0.0, k.x) * h_tilde;
            slope_y_field[i][j] = Complex32::new(0.0, k.y) * h_tilde;
            
            // Displacement field for choppy waves (Eq. 29)
            disp_x_field[i][j] = Complex32::new(0.0, -k_hat.x) * h_tilde * params.choppiness;
            disp_y_field[i][j] = Complex32::new(0.0, -k_hat.y) * h_tilde * params.choppiness;
        }
    }
    
    // Prepare FFT
    let mut planner = FftPlanner::new();
    
    // Perform inverse FFT to get spatial representation
    let mut height = ifft2d(&mut planner, height_field);
    let slope_x = ifft2d(&mut planner, slope_x_field);
    let slope_y = ifft2d(&mut planner, slope_y_field);
    let disp_x = ifft2d(&mut planner, disp_x_field);
    let disp_y = ifft2d(&mut planner, disp_y_field);
    
    // First pass: calculate the current min/max to determine actual range
    let mut min_height = f32::MAX;
    let mut max_height = f32::MIN;
    
    for i in 0..n {
        for j in 0..n {
            min_height = min_height.min(height[i][j]);
            max_height = max_height.max(height[i][j]);
        }
    }
    
    // Calculate current range and target range
    let current_range = max_height - min_height;
    let target_range = params.amplitude * 2.0; // We want peak-to-peak to be 2*amplitude
    
    // Calculate scaling factor to achieve the target range
    // Avoid division by zero
    let height_scale_factor = if current_range > 1e-6 {
        target_range / current_range
    } else {
        // If there's no variation, use a reasonable default
        params.amplitude
    };
    
    println!("Wave height scaling: Raw range = {:.6}, Target = {:.6}, Scale factor = {:.6}", 
             current_range, target_range, height_scale_factor);
    
    // Apply scaling to match the commanded amplitude
    for i in 0..n {
        for j in 0..n {
            // Center around zero and scale to match the target amplitude
            height[i][j] = (height[i][j] - (min_height + max_height) / 2.0) * height_scale_factor;
        }
    }
    
    // Compute normal vectors from slope
    let mut normal_x = vec![vec![0.0; n]; n];
    let mut normal_y = vec![vec![0.0; n]; n];
    let mut normal_z = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            // Create normal from slope (normalized)
            let nx = -slope_x[i][j];
            let ny = -slope_y[i][j];
            let nz = 1.0;

            // Normalize
            let length = (nx * nx + ny * ny + nz * nz).sqrt();
            normal_x[i][j] = nx / length;
            normal_y[i][j] = ny / length;
            normal_z[i][j] = nz / length;
        }
    }
    
    // Create and return the displacement field
    DisplacementField {
        resolution: n,
        height,
        displacement_x: disp_x,
        displacement_y: disp_y,
        normal_x,
        normal_y,
        normal_z,
    }
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
