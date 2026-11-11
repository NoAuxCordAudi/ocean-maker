//! GLB export, either a static surface or a morph-target animation.
//!
//! Tiles are nodes that share a single mesh (instancing), so an N×N tiling costs N² small
//! nodes, not N² copies of the vertex data. In animated exports every tile node gets its own
//! weights channel, all driven by one shared sampler.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use sha2::{Digest, Sha256};
use mesh_tools::material::MaterialBuilder;
use mesh_tools::mesh::MeshBuilder;
use mesh_tools::texture::TextureFormat;
use mesh_tools::{
    AccessorSpec, AccessorType, AnimationPath, BufferEncoding, Bounds, ExtensionTarget, GltfBuilder,
    InterpolationType, MagFilter, MinFilter, MorphTarget, SamplerDesc, TextureSlot, WrapMode,
};
use serde_json::{json, Value};

use crate::atlas::{encode_rgb_png, render_atlas, AtlasLayout};
use crate::foam::{FoamParams, FoamSequence};
use crate::mesh::{GridSpec, MeshData};

/// Morph-target animation: `frames` are absolute surfaces sampled at `fps`.
#[derive(Debug, Clone, Copy)]
pub struct Animation<'a> {
    pub fps: f32,
    pub frames: &'a [MeshData],
}

/// Estimated size in bytes of the morph data for `frame_count` frames of `vertex_count` vertices
/// (position and normal deltas, f32 VEC3 each).
pub fn morph_data_bytes(vertex_count: usize, frame_count: usize) -> usize {
    vertex_count * frame_count * 2 * 3 * std::mem::size_of::<f32>()
}

fn flatten(v: impl Iterator<Item = glam::Vec3>) -> Vec<f32> {
    v.flat_map(|p| p.to_array()).collect()
}

/// One morph target per frame: deltas from the rest pose `base`.
fn morph_targets(base: &MeshData, frames: &[MeshData]) -> Result<Vec<MorphTarget>> {
    frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            ensure!(
                f.positions.len() == base.positions.len() && f.normals.len() == base.normals.len(),
                "frame {i} has {} vertices but the base mesh has {}",
                f.positions.len(),
                base.positions.len()
            );
            let positions = flatten(f.positions.iter().zip(&base.positions).map(|(a, b)| *a - *b));
            let normals = flatten(f.normals.iter().zip(&base.normals).map(|(a, b)| *a - *b));
            Ok(MorphTarget::new().with_positions(positions).with_normals(normals))
        })
        .collect()
}

/// Write `base` (plus an optional morph-target `animation`) as a GLB with `tiles`×`tiles`
/// instances placed one grid period apart.
///
/// For an animation `base` is the rest pose (typically [`crate::mesh::flat_grid`]) and each
/// frame becomes a target with weight 1 at its keyframe time `i / fps`, linearly blended.
pub fn export_glb(
    base: &MeshData,
    animation: Option<Animation<'_>>,
    grid: &GridSpec,
    tiles: usize,
    output_path: &Path,
) -> Result<()> {
    export_glb_with(base, animation, None, grid, tiles, output_path)
}

/// Linear-light colours baked into the foam atlas.
const WATER_COLOR: [f32; 3] = [0.0, 0.3, 0.8];
const FOAM_COLOR: [f32; 3] = [0.9, 0.9, 0.9];
const FOAM_ROUGHNESS: f32 = 0.35;

/// Like [`export_glb`], and with `foam` the foam is embedded in the GLB (one self-contained file).
///
/// All foam frames are baked into one base-colour atlas texture (see [`crate::atlas`]); the
/// material's `KHR_texture_transform` offset selects a frame and, for an animation, a
/// `KHR_animation_pointer` channel (STEP) steps the offset in sync with the morph-target weights.
/// Viewers without pointer-animation support show the first frame's foam statically. Foam needs
/// the same number of frames as the animation (1 for a still).
pub fn export_glb_with(
    base: &MeshData,
    animation: Option<Animation<'_>>,
    foam: Option<&FoamSequence>,
    grid: &GridSpec,
    tiles: usize,
    output_path: &Path,
) -> Result<()> {
    ensure!(tiles >= 1, "tiles must be at least 1");
    if let Some(f) = foam {
        let want = animation.as_ref().map_or(1, |a| a.frames.len());
        ensure!(f.frames.len() == want, "foam has {} frames but the animation has {want}", f.frames.len());
    }
    let path_str = output_path
        .to_str()
        .ok_or_else(|| anyhow!("output path is not valid UTF-8: {}", output_path.display()))?;

    let mut builder = GltfBuilder::new();
    let (material, atlas) = match foam {
        None => (builder.create_basic_material(Some("OceanMaterial".to_string()), [0.0, 0.3, 0.8, 1.0]), None),
        Some(f) => {
            let (material, layout) = add_foam_material(&mut builder, f)?;
            (material, Some(layout))
        }
    };

    let mut mesh_builder = MeshBuilder::new(Some("OceanSurface".to_string()))
        .with_positions(flatten(base.positions.iter().copied()))
        .with_indices_u32(base.indices.clone())
        .with_normals(flatten(base.normals.iter().copied()))
        .with_texcoords(base.uvs.iter().flat_map(|uv| uv.to_array()).collect())
        .with_material(material);

    if let Some(anim) = &animation {
        ensure!(anim.fps.is_finite() && anim.fps > 0.0, "fps must be > 0, got {}", anim.fps);
        ensure!(!anim.frames.is_empty(), "animation needs at least one frame");
        let digits = (anim.frames.len() - 1).to_string().len().max(3);
        let names = (0..anim.frames.len()).map(|i| format!("frame_{i:0digits$}")).collect();
        mesh_builder = mesh_builder
            .with_morph_targets(morph_targets(base, anim.frames)?)
            .with_target_names(names);
    }
    let mesh_index = mesh_builder.build_into(&mut builder).context("failed to build ocean mesh")?;

    // Tile placement: odd counts are centred on the origin; even counts extend one tile
    // further in the negative direction.
    let half = tiles as isize / 2;
    let end = if tiles.is_multiple_of(2) { half - 1 } else { half };
    let mut node_indices = Vec::with_capacity(tiles * tiles);
    for row in -half..=end {
        for col in -half..=end {
            let translation = (row != 0 || col != 0)
                .then(|| [col as f32 * grid.width(), row as f32 * grid.length(), 0.0]);
            node_indices.push(builder.add_node(
                Some(format!("OceanTile_{row}_{col}")),
                Some(mesh_index),
                translation,
                None,
                None,
            ));
        }
    }

    if let Some(anim) = &animation {
        let n = anim.frames.len();
        let times: Vec<f32> = (0..n).map(|i| i as f32 / anim.fps).collect();
        // One-hot weights: keyframe i activates target i only.
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();

        let animation_index = builder.add_animation(Some("OceanWaves".to_string()));
        let (_, sampler) = builder
            .add_weights_track(animation_index, node_indices[0], &times, &weights, InterpolationType::Linear)
            .context("failed to add weights animation")?;
        for &node in &node_indices[1..] {
            builder
                .add_animation_channel(animation_index, sampler, node, AnimationPath::Weights)
                .context("failed to animate tile")?;
        }
    }

    // Offset animation (needs patching into the JSON after validation, see below).
    let mut pointer_channel = None;
    if let (Some(layout), Some(anim)) = (&atlas, &animation) {
        let times: Vec<f32> = (0..anim.frames.len()).map(|i| i as f32 / anim.fps).collect();
        let offsets: Vec<f32> = (0..anim.frames.len()).flat_map(|i| layout.uv_transform(i).0).collect();
        let input = builder
            .add_accessor_from_slice(&times, AccessorType::Scalar, AccessorSpec::new().with_bounds(Bounds::Compute))?;
        let output = builder.add_accessor_from_slice(&offsets, AccessorType::Vec2, AccessorSpec::new())?;
        let animation_index = builder.gltf.animations.as_ref().map_or(0, |a| a.len()) - 1;
        let sampler = builder.add_animation_sampler(animation_index, input, output, InterpolationType::Step)?;
        pointer_channel = Some((animation_index, sampler, material));
    }

    builder.add_scene(Some("Ocean Tiles Scene".to_string()), Some(node_indices));
    match pointer_channel {
        None => builder.export_glb(path_str)?,
        Some((animation_index, sampler, material)) => {
            builder.declare_extension_used("KHR_animation_pointer")?;
            let parts = builder.to_gltf_parts(&BufferEncoding::Glb)?;
            let mut doc: Value = serde_json::from_str(&parts.json)?;
            add_pointer_channel(&mut doc, animation_index, sampler, material)?;
            fs::write(output_path, glb_container(&serde_json::to_string(&doc)?, &parts.bin))
                .with_context(|| format!("cannot write {}", output_path.display()))?;
        }
    }
    Ok(())
}

/// Embed the foam atlas and create the base-colour-textured material; returns the material index.
fn add_foam_material(builder: &mut GltfBuilder, foam: &FoamSequence) -> Result<(usize, AtlasLayout)> {
    let layout = AtlasLayout::new(foam.frames.len(), foam.resolution)?;
    let pixels = render_atlas(&layout, &foam.frames, WATER_COLOR, FOAM_COLOR)?;
    let png = encode_rgb_png(&pixels, layout.width(), layout.height())?;

    let image = builder.add_image_bytes(Some("FoamAtlas".to_string()), TextureFormat::PNG, &png)?;
    // Cells are separated by wrapped gutters, so clamp (never wrap into another cell) and keep
    // filtering linear without mipmaps (a mip chain would blend neighbouring cells).
    let sampler = builder.add_sampler(SamplerDesc {
        mag_filter: Some(MagFilter::Linear),
        min_filter: Some(MinFilter::Linear),
        wrap_s: Some(WrapMode::ClampToEdge),
        wrap_t: Some(WrapMode::ClampToEdge),
    });
    let texture = builder.add_texture(Some("FoamAtlas".to_string()), image, Some(sampler))?;
    let material = builder.add_material_from(
        MaterialBuilder::new(Some("OceanFoamMaterial".to_string()))
            .with_base_color_texture(texture)
            .with_metallic_factor(0.0)
            .with_roughness_factor(FOAM_ROUGHNESS)
            .build(),
    )?;

    let (offset, scale) = layout.uv_transform(0);
    builder.set_extension(
        ExtensionTarget::MaterialTexture { material, slot: TextureSlot::BaseColor },
        "KHR_texture_transform",
        json!({ "offset": offset, "scale": scale }),
    )?;
    Ok((material, layout))
}

/// Append a `KHR_animation_pointer` channel driving the base-colour texture offset.
///
/// Pointer channels must not have a `target.node`, which the typed builder always writes, hence
/// the JSON patch.
fn add_pointer_channel(doc: &mut Value, animation: usize, sampler: usize, material: usize) -> Result<()> {
    let channels = doc
        .pointer_mut(&format!("/animations/{animation}"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow!("animation {animation} missing from the document"))?
        .entry("channels")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| anyhow!("animation channels is not an array"))?;
    channels.push(json!({
        "sampler": sampler,
        "target": {
            "path": "pointer",
            "extensions": { "KHR_animation_pointer": {
                "pointer": format!(
                    "/materials/{material}/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset"
                )
            } }
        }
    }));
    Ok(())
}

/// Assemble a GLB container: JSON chunk (space-padded) then BIN chunk (zero-padded).
fn glb_container(json: &str, bin: &[u8]) -> Vec<u8> {
    let pad = |n: usize| (4 - n % 4) % 4;
    let (jp, bp) = (pad(json.len()), pad(bin.len()));
    let mut out = Vec::with_capacity(28 + json.len() + jp + bin.len() + bp);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&[0; 4]); // total length, patched below
    out.extend_from_slice(&((json.len() + jp) as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    out.resize(out.len() + jp, 0x20);
    if !bin.is_empty() {
        out.extend_from_slice(&((bin.len() + bp) as u32).to_le_bytes());
        out.extend_from_slice(&0x004E4942u32.to_le_bytes());
        out.extend_from_slice(bin);
        out.resize(out.len() + bp, 0);
    }
    let total = out.len() as u32;
    out[8..12].copy_from_slice(&total.to_le_bytes());
    out
}

/// Companion Blender script, written next to the foam images.
const BLENDER_SCRIPT: &str = include_str!("blender/ocean_blender.py");

/// Everything needed to write the foam sidecar files.
#[derive(Debug, Clone, Copy)]
pub struct FoamExport<'a> {
    pub sequence: &'a FoamSequence,
    pub params: &'a FoamParams,
    /// Frame rate of the GLB animation (also used to time the foam frames).
    pub fps: f32,
    /// The GLB the foam belongs to; must already exist.
    pub glb_path: &'a Path,
    /// Number of shape keys (morph targets) the GLB has; 0 for a still.
    pub shape_keys: usize,
    pub write_blender_script: bool,
    /// Ask the Blender script to break the mask up with procedural noise.
    pub noise_breakup: bool,
}

/// Encode a foam mask as a 16-bit grayscale PNG. `mask` is row-major, `resolution`² values in
/// [0, 1] (clamped); row 0 is the top of the image (glTF `v` ~ 0).
pub fn encode_foam_png(mask: &[f32], resolution: usize) -> Result<Vec<u8>> {
    ensure!(mask.len() == resolution * resolution, "mask has {} values, expected {}", mask.len(), resolution * resolution);
    ensure!(mask.iter().all(|v| v.is_finite()), "foam mask contains a non-finite value");
    let mut bytes = Vec::with_capacity(mask.len() * 2);
    for &v in mask {
        let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
        bytes.extend_from_slice(&q.to_be_bytes());
    }
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, resolution as u32, resolution as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&bytes)?;
    writer.finish()?;
    Ok(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Path of the GLB as the manifest should reference it: relative to `dir` when it lives in `dir`
/// or its parent, otherwise absolute.
fn glb_reference(dir: &Path, glb: &Path) -> Result<String> {
    let dir = fs::canonicalize(dir).with_context(|| format!("cannot resolve {}", dir.display()))?;
    let glb = fs::canonicalize(glb).with_context(|| format!("GLB not found: {}", glb.display()))?;
    let name = glb.file_name().and_then(|n| n.to_str()).ok_or_else(|| anyhow!("GLB path is not valid UTF-8"))?;
    let parent = glb.parent();
    Ok(if parent == Some(dir.as_path()) {
        name.to_string()
    } else if parent == dir.parent() {
        format!("../{name}")
    } else {
        glb.to_str().ok_or_else(|| anyhow!("GLB path is not valid UTF-8"))?.to_string()
    })
}

/// Write `foam_NNNN.png` for every frame, `foam.json` (with SHA-256 of each image) and, unless
/// disabled, `ocean_blender.py` into `dir`.
///
/// Texel centres are at `((col + 0.5) / M, (row + 0.5) / M)` of the patch and the images need no
/// flip to match the GLB's UVs.
pub fn write_foam_sequence(dir: &Path, e: &FoamExport<'_>) -> Result<()> {
    let seq = e.sequence;
    ensure!(!seq.frames.is_empty(), "foam sequence has no frames");
    ensure!(e.fps.is_finite() && e.fps > 0.0, "fps must be > 0, got {}", e.fps);
    fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;

    let digits = (seq.frames.len() - 1).to_string().len().max(4);
    let mut files = Vec::with_capacity(seq.frames.len());
    for (i, frame) in seq.frames.iter().enumerate() {
        let bytes = encode_foam_png(frame, seq.resolution).with_context(|| format!("foam frame {i}"))?;
        let name = format!("foam_{i:0digits$}.png");
        fs::write(dir.join(&name), &bytes).with_context(|| format!("cannot write {name}"))?;
        files.push(serde_json::json!({ "file": name, "sha256": sha256_hex(&bytes) }));
    }

    let manifest = serde_json::json!({
        "version": 1,
        "glb": glb_reference(dir, e.glb_path)?,
        "fps": e.fps,
        "frame_count": seq.frames.len(),
        "shape_keys": e.shape_keys,
        "noise_breakup": e.noise_breakup,
        "resolution": seq.resolution,
        "image": "16-bit grayscale PNG, row 0 = top (glTF v ~ 0), texel centres at (i + 0.5) / resolution; non-colour data, repeat",
        "foam": {
            "threshold": seq.threshold,
            "trigger_fraction": seq.trigger_fraction,
            "mean_foam": seq.mean_foam,
            "spinup_frames": seq.spinup_frames,
            "spinup_residual": seq.residual,
            "tau": e.params.tau,
            "old_tau": e.params.old_tau,
            "old_weight": e.params.old_weight,
            "softness": e.params.softness,
        },
        "foam_files": files,
    });
    fs::write(dir.join("foam.json"), serde_json::to_string_pretty(&manifest)? + "\n")?;
    if e.write_blender_script {
        fs::write(dir.join("ocean_blender.py"), BLENDER_SCRIPT)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{flat_grid, ocean_mesh};
    use crate::tessendorf::{compute_wave_fields, frame_times, OceanParams};

    struct Fixture {
        grid: GridSpec,
        base: MeshData,
        frames: Vec<MeshData>,
        times: Vec<f32>,
    }

    fn fixture(n: usize) -> Fixture {
        let size = 10.0;
        let params = OceanParams { resolution: n, size, choppiness: 0.6, ..Default::default() };
        let times = frame_times(0.5, 1.0, 4.0).unwrap();
        let fields = compute_wave_fields(&params, &times).unwrap();
        let grid = GridSpec::new(size, size, n, n).unwrap();
        let frames = fields.iter().map(|f| ocean_mesh(&grid, size, f)).collect();
        Fixture { base: flat_grid(&grid), grid, frames, times }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ocean_export_test_{}_{name}.glb", std::process::id()))
    }

    #[test]
    fn animated_export_round_trips() {
        let fx = fixture(8);
        let path = tmp("anim");
        let anim = Animation { fps: 4.0, frames: &fx.frames };
        export_glb(&fx.base, Some(anim), &fx.grid, 1, &path).unwrap();

        let (doc, buffers, _) = gltf::import(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let mesh = doc.meshes().next().unwrap();
        let prim = mesh.primitives().next().unwrap();
        let targets: Vec<_> = prim.morph_targets().collect();
        assert_eq!(targets.len(), fx.frames.len());

        // base + delta reconstructs each frame's absolute positions
        let reader = prim.reader(|b| Some(&buffers[b.index()]));
        let base: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
        assert!(base.iter().all(|p| p[2] == 0.0), "base must be the flat rest pose");
        for (t, frame) in reader.read_morph_targets().zip(&fx.frames) {
            let deltas: Vec<[f32; 3]> = t.0.unwrap().collect();
            for ((b, d), want) in base.iter().zip(&deltas).zip(&frame.positions) {
                for k in 0..3 {
                    assert!((b[k] + d[k] - want[k]).abs() < 1e-5);
                }
            }
        }

        // keyframe times are i / fps and weights are one-hot
        let anim = doc.animations().next().unwrap();
        let channel = anim.channels().next().unwrap();
        let r = channel.reader(|b| Some(&buffers[b.index()]));
        let times: Vec<f32> = r.read_inputs().unwrap().collect();
        let want: Vec<f32> = (0..fx.frames.len()).map(|i| i as f32 / 4.0).collect();
        assert_eq!(times, want);
        let gltf::animation::util::ReadOutputs::MorphTargetWeights(w) = r.read_outputs().unwrap() else {
            panic!("expected weights output");
        };
        let w: Vec<f32> = w.into_f32().collect();
        let n = fx.frames.len();
        assert_eq!(w.len(), n * n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(w[i * n + j], if i == j { 1.0 } else { 0.0 });
            }
        }
        assert_eq!(fx.times.len(), n);
    }

    #[test]
    fn tiles_share_one_mesh_and_all_animate() {
        let fx = fixture(8);
        let anim = Animation { fps: 4.0, frames: &fx.frames };
        let p1 = tmp("t1");
        let p3 = tmp("t3");
        export_glb(&fx.base, Some(anim), &fx.grid, 1, &p1).unwrap();
        export_glb(&fx.base, Some(anim), &fx.grid, 3, &p3).unwrap();

        let (doc, _, _) = gltf::import(&p3).unwrap();
        let (s1, s3) = (std::fs::metadata(&p1).unwrap().len(), std::fs::metadata(&p3).unwrap().len());
        std::fs::remove_file(&p1).unwrap();
        std::fs::remove_file(&p3).unwrap();

        assert_eq!(doc.meshes().count(), 1);
        let nodes: Vec<_> = doc.nodes().collect();
        assert_eq!(nodes.len(), 9);
        assert!(nodes.iter().all(|n| n.mesh().unwrap().index() == 0));
        let anim = doc.animations().next().unwrap();
        assert_eq!(anim.channels().count(), 9);
        assert_eq!(anim.samplers().count(), 1, "tiles must share one sampler");
        // 9 tiles add only node/channel JSON, not vertex data.
        assert!(s3 < s1 + 4096, "3x3 file ({s3}) should be about the size of 1x1 ({s1})");
    }

    #[test]
    fn static_export_has_no_animation() {
        let fx = fixture(8);
        let path = tmp("static");
        export_glb(&fx.frames[0], None, &fx.grid, 2, &path).unwrap();
        let (doc, _, _) = gltf::import(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(doc.animations().count(), 0);
        assert_eq!(doc.meshes().count(), 1);
        assert_eq!(doc.nodes().count(), 4);
    }

    #[test]
    fn rejects_bad_input() {
        let fx = fixture(8);
        let path = tmp("bad");
        assert!(export_glb(&fx.base, None, &fx.grid, 0, &path).is_err());
        let bad_fps = Animation { fps: 0.0, frames: &fx.frames };
        assert!(export_glb(&fx.base, Some(bad_fps), &fx.grid, 1, &path).is_err());
        let empty = Animation { fps: 24.0, frames: &[] };
        assert!(export_glb(&fx.base, Some(empty), &fx.grid, 1, &path).is_err());
        // frame with the wrong vertex count
        let other = fixture(4);
        let mismatched = Animation { fps: 24.0, frames: &other.frames };
        assert!(export_glb(&fx.base, Some(mismatched), &fx.grid, 1, &path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn morph_size_estimate() {
        assert_eq!(morph_data_bytes(100, 10), 100 * 10 * 24);
    }

    // ---- foam sidecar ----

    use crate::foam::FoamSequence;

    fn sequence(frames: Vec<Vec<f32>>, m: usize) -> FoamSequence {
        FoamSequence {
            resolution: m,
            threshold: 0.7,
            spinup_frames: 3,
            residual: 0.01,
            trigger_fraction: 0.02,
            mean_foam: 0.1,
            frames,
        }
    }

    fn decode(path: &Path) -> (u32, Vec<u16>) {
        let decoder = png::Decoder::new(fs::File::open(path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.color_type, png::ColorType::Grayscale);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        let vals = buf[..info.buffer_size()].chunks(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        (info.width, vals)
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ocean_foam_test_{}_{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn foam_png_keeps_orientation_and_values() {
        let m = 8;
        let mut mask = vec![0.0f32; m * m];
        mask[m + 3] = 1.0; // col 3, row 1 (second row from the top)
        mask[6 * m + 5] = 0.5;
        let dir = scratch("png");
        let path = dir.join("a.png");
        fs::write(&path, encode_foam_png(&mask, m).unwrap()).unwrap();
        let (w, px) = decode(&path);
        assert_eq!(w as usize, m);
        assert_eq!(px[m + 3], 65535);
        assert_eq!(px[6 * m + 5], 32768); // round(0.5 * 65535)
        assert_eq!(px.iter().filter(|&&v| v != 0).count(), 2);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn foam_png_is_deterministic_and_validates() {
        let mask: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        assert_eq!(encode_foam_png(&mask, 4).unwrap(), encode_foam_png(&mask, 4).unwrap());
        assert!(encode_foam_png(&mask, 5).is_err());
        assert!(encode_foam_png(&[f32::NAN; 4], 2).is_err());
        // out-of-range values are clamped, not wrapped
        let clamped = encode_foam_png(&[-1.0, 2.0, 0.0, 1.0], 2).unwrap();
        let exact = encode_foam_png(&[0.0, 1.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(clamped, exact);
    }

    #[test]
    fn foam_sidecar_manifest_hashes_match_files() {
        let fx = fixture(8);
        let dir = scratch("sidecar");
        let glb = dir.join("ocean.glb");
        let anim = Animation { fps: 4.0, frames: &fx.frames };
        export_glb(&fx.base, Some(anim), &fx.grid, 1, &glb).unwrap();

        let frames: Vec<Vec<f32>> = (0..fx.frames.len()).map(|f| vec![f as f32 / 10.0; 16]).collect();
        let seq = sequence(frames, 4);
        let params = FoamParams::default();
        let foam_dir = dir.join("ocean_foam");
        let e = FoamExport {
            sequence: &seq, params: &params, fps: 4.0, glb_path: &glb,
            shape_keys: fx.frames.len(), write_blender_script: true, noise_breakup: false,
        };
        write_foam_sequence(&foam_dir, &e).unwrap();

        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(foam_dir.join("foam.json")).unwrap()).unwrap();
        assert_eq!(manifest["glb"], "../ocean.glb");
        assert_eq!(manifest["frame_count"], fx.frames.len());
        assert_eq!(manifest["shape_keys"], fx.frames.len());
        assert_eq!(manifest["noise_breakup"], false);
        let files = manifest["foam_files"].as_array().unwrap();
        assert_eq!(files.len(), fx.frames.len());
        for f in files {
            let bytes = fs::read(foam_dir.join(f["file"].as_str().unwrap())).unwrap();
            assert_eq!(f["sha256"], sha256_hex(&bytes));
        }
        assert_eq!(files[0]["file"], "foam_0000.png");
        assert_eq!(fs::read_to_string(foam_dir.join("ocean_blender.py")).unwrap(), BLENDER_SCRIPT);

        // regenerating gives byte-identical output
        let first = fs::read(foam_dir.join("foam.json")).unwrap();
        write_foam_sequence(&foam_dir, &e).unwrap();
        assert_eq!(first, fs::read(foam_dir.join("foam.json")).unwrap());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn foam_sidecar_can_skip_script_and_rejects_bad_input() {
        let fx = fixture(8);
        let dir = scratch("noscript");
        let glb = dir.join("o.glb");
        export_glb(&fx.frames[0], None, &fx.grid, 1, &glb).unwrap();
        let seq = sequence(vec![vec![0.5; 16]], 4);
        let params = FoamParams::default();
        fn mk<'a>(seq: &'a FoamSequence, params: &'a FoamParams, fps: f32, glb: &'a Path, script: bool) -> FoamExport<'a> {
            FoamExport { sequence: seq, params, fps, glb_path: glb, shape_keys: 0, write_blender_script: script, noise_breakup: true }
        }
        write_foam_sequence(&dir, &mk(&seq, &params, 24.0, &glb, false)).unwrap();
        assert!(!dir.join("ocean_blender.py").exists());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fs::read_to_string(dir.join("foam.json")).unwrap()).unwrap()["glb"],
            "o.glb"
        );
        assert!(write_foam_sequence(&dir, &mk(&seq, &params, 0.0, &glb, false)).is_err());
        assert!(write_foam_sequence(&dir, &mk(&seq, &params, 24.0, &dir.join("missing.glb"), false)).is_err());
        let empty = sequence(vec![], 4);
        assert!(write_foam_sequence(&dir, &mk(&empty, &params, 24.0, &glb, false)).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn blender_script_is_self_contained() {
        assert!(BLENDER_SCRIPT.contains("def load_and_verify"));
        assert!(BLENDER_SCRIPT.contains("import bpy  # only available inside Blender"));
    }

    // ---- embedded foam atlas (one-file GLB) ----

    /// Parse a GLB container into (JSON, BIN), checking the framing.
    fn read_glb(path: &Path) -> (Value, Vec<u8>) {
        let bytes = fs::read(path).unwrap();
        assert_eq!(&bytes[..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, bytes.len());
        let jl = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(&bytes[16..20], b"JSON");
        assert_eq!(jl % 4, 0);
        let json: Value = serde_json::from_slice(&bytes[20..20 + jl]).unwrap();
        let rest = &bytes[20 + jl..];
        let bl = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        assert_eq!(&rest[4..8], b"BIN\0");
        assert_eq!(bl % 4, 0);
        assert_eq!(rest.len(), 8 + bl);
        (json, rest[8..].to_vec())
    }

    fn floats(doc: &Value, bin: &[u8], accessor: usize) -> (String, usize, Vec<f32>) {
        let a = &doc["accessors"][accessor];
        assert_eq!(a["componentType"], 5126);
        let view = &doc["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
        let start = view["byteOffset"].as_u64().unwrap_or(0) as usize + a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let count = a["count"].as_u64().unwrap() as usize;
        let comps = match a["type"].as_str().unwrap() { "SCALAR" => 1, "VEC2" => 2, "VEC3" => 3, t => panic!("{t}") };
        let vals = (0..count * comps)
            .map(|i| f32::from_le_bytes(bin[start + 4 * i..start + 4 * i + 4].try_into().unwrap()))
            .collect();
        (a["type"].as_str().unwrap().to_string(), count, vals)
    }

    fn foam_for(n: usize) -> FoamSequence {
        let frames = (0..n).map(|f| (0..16).map(|i| ((i + f) % 5) as f32 / 4.0).collect()).collect();
        sequence(frames, 4)
    }

    #[test]
    fn embedded_foam_animation_has_atlas_transform_and_pointer_channel() {
        let fx = fixture(8);
        let n = fx.frames.len();
        let foam = foam_for(n);
        let path = tmp("atlas_anim");
        let anim = Animation { fps: 4.0, frames: &fx.frames };
        export_glb_with(&fx.base, Some(anim), Some(&foam), &fx.grid, 3, &path).unwrap();
        let (doc, bin) = read_glb(&path);
        fs::remove_file(&path).unwrap();

        let used: Vec<&str> = doc["extensionsUsed"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert!(used.contains(&"KHR_texture_transform") && used.contains(&"KHR_animation_pointer"), "{used:?}");
        assert!(doc.get("extensionsRequired").is_none(), "extensions must stay optional");

        // one embedded PNG image of the atlas size
        let layout = AtlasLayout::new(n, 4).unwrap();
        assert_eq!(doc["images"].as_array().unwrap().len(), 1);
        assert_eq!(doc["images"][0]["mimeType"], "image/png");
        let view = &doc["bufferViews"][doc["images"][0]["bufferView"].as_u64().unwrap() as usize];
        let (off, len) = (view["byteOffset"].as_u64().unwrap_or(0) as usize, view["byteLength"].as_u64().unwrap() as usize);
        let reader = png::Decoder::new(&bin[off..off + len]).read_info().unwrap();
        assert_eq!((reader.info().width as usize, reader.info().height as usize), (layout.width(), layout.height()));

        // sampler clamps (no wrapping into neighbouring cells)
        assert_eq!(doc["samplers"][0]["wrapS"], 33071);
        assert_eq!(doc["samplers"][0]["wrapT"], 33071);

        // material uses the texture with frame 0's transform
        let bct = &doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"];
        let (o0, s0) = layout.uv_transform(0);
        let tt = &bct["extensions"]["KHR_texture_transform"];
        assert_eq!(tt["offset"], json!(o0));
        assert_eq!(tt["scale"], json!(s0));
        assert_eq!(doc["materials"].as_array().unwrap().len(), 1);

        // exactly one pointer channel (no node), STEP, VEC2 offsets that match the layout
        let channels = doc["animations"][0]["channels"].as_array().unwrap();
        let pointers: Vec<_> = channels.iter().filter(|c| c["target"]["path"] == "pointer").collect();
        assert_eq!(pointers.len(), 1);
        assert_eq!(channels.len(), 9 + 1, "9 tile weight channels + 1 pointer channel");
        let target = &pointers[0]["target"];
        assert!(target.get("node").is_none(), "pointer channels must not name a node");
        assert_eq!(
            target["extensions"]["KHR_animation_pointer"]["pointer"],
            "/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset"
        );
        let sampler = &doc["animations"][0]["samplers"][pointers[0]["sampler"].as_u64().unwrap() as usize];
        assert_eq!(sampler["interpolation"], "STEP");
        let (ty, count, offs) = floats(&doc, &bin, sampler["output"].as_u64().unwrap() as usize);
        assert_eq!((ty.as_str(), count), ("VEC2", n));
        let want: Vec<f32> = (0..n).flat_map(|i| layout.uv_transform(i).0).collect();
        assert_eq!(offs, want);
        let (_, tc, times) = floats(&doc, &bin, sampler["input"].as_u64().unwrap() as usize);
        assert_eq!(tc, n);
        assert_eq!(times, (0..n).map(|i| i as f32 / 4.0).collect::<Vec<_>>());
        // morph weights channels are untouched
        assert_eq!(channels.iter().filter(|c| c["target"]["path"] == "weights").count(), 9);
    }

    #[test]
    fn embedded_foam_still_has_static_transform_and_no_pointer() {
        let fx = fixture(8);
        let foam = foam_for(1);
        let path = tmp("atlas_still");
        export_glb_with(&fx.frames[0], None, Some(&foam), &fx.grid, 2, &path).unwrap();
        let (doc, _) = read_glb(&path);
        fs::remove_file(&path).unwrap();
        assert!(doc.get("animations").is_none());
        let used = doc["extensionsUsed"].as_array().unwrap();
        assert!(used.iter().any(|v| v == "KHR_texture_transform"));
        assert!(!used.iter().any(|v| v == "KHR_animation_pointer"));
        assert!(doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["extensions"]["KHR_texture_transform"]["offset"].is_array());
    }

    #[test]
    fn without_foam_the_material_stays_untextured() {
        let fx = fixture(8);
        let path = tmp("nofoam");
        export_glb(&fx.frames[0], None, &fx.grid, 1, &path).unwrap();
        let (doc, _) = read_glb(&path);
        fs::remove_file(&path).unwrap();
        assert!(doc.get("images").is_none());
        assert!(doc["materials"][0]["pbrMetallicRoughness"].get("baseColorTexture").is_none());
    }

    #[test]
    fn embedded_foam_rejects_mismatched_frames() {
        let fx = fixture(8);
        let path = tmp("atlas_bad");
        let anim = Animation { fps: 4.0, frames: &fx.frames };
        let err = export_glb_with(&fx.base, Some(anim), Some(&foam_for(2)), &fx.grid, 1, &path).unwrap_err();
        assert!(err.to_string().contains("foam has 2 frames"), "{err}");
        assert!(export_glb_with(&fx.frames[0], None, Some(&foam_for(3)), &fx.grid, 1, &path).is_err());
        assert!(!path.exists());
    }
}
