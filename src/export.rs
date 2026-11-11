//! GLB export, either a static surface or a morph-target animation.
//!
//! Tiles are nodes that share a single mesh (instancing), so an N×N tiling costs N² small
//! nodes, not N² copies of the vertex data. In animated exports every tile node gets its own
//! weights channel, all driven by one shared sampler.

use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use mesh_tools::mesh::MeshBuilder;
use mesh_tools::{AnimationPath, GltfBuilder, InterpolationType, MorphTarget};

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
    ensure!(tiles >= 1, "tiles must be at least 1");
    let path_str = output_path
        .to_str()
        .ok_or_else(|| anyhow!("output path is not valid UTF-8: {}", output_path.display()))?;

    let mut builder = GltfBuilder::new();
    let material = builder.create_basic_material(Some("OceanMaterial".to_string()), [0.0, 0.3, 0.8, 1.0]);

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

    builder.add_scene(Some("Ocean Tiles Scene".to_string()), Some(node_indices));
    builder.export_glb(path_str)?;
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
}
