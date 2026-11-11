//! Tessendorf FFT ocean surface generation.
//!
//! - [`tessendorf`]: spectrum, wave-field simulation and periodic sampling
//! - [`mesh`]: tileable grid meshes built from a wave field
//! - [`export`]: GLB export, static or animated with morph targets

pub mod export;
pub mod mesh;
pub mod tessendorf;
