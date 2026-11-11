//! Tessendorf FFT ocean surface generation.
//!
//! - [`tessendorf`]: spectrum, wave-field simulation and periodic sampling
//! - [`foam`]: whitecap foam masks from the displacement Jacobian
//! - [`mesh`]: tileable grid meshes built from a wave field
//! - [`export`]: GLB export, static or animated with morph targets

pub mod atlas;
pub mod export;
pub mod foam;
pub mod mesh;
pub mod tessendorf;
