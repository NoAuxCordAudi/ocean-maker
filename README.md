# Ocean Generator

Library and CLI that generate a tileable ocean surface with Tessendorf's FFT wave model and
export it as GLB — a single still frame, or an animation stored as glTF morph targets (shape
keys in Blender).

![Ocean Animation](ocean_animation.gif)

## Build

```bash
cargo build --release   # binary: target/release/ocean-generator
cargo test
```

## CLI

```bash
# Still frame, 3x3 instanced tiles
ocean-generator -g 3 -o ocean.glb

# Wind from 45 degrees, bigger waves
ocean-generator --wind-angle 45 --wind-speed 8 -a 1.5 -o ocean.glb

# 5 second animation at 24 fps as morph targets
ocean-generator --anim-duration 5 --anim-fps 24 -g 3 -o ocean_anim.glb
```

| Option | Default | Meaning |
|---|---|---|
| `-w, --width` / `-l, --length` | 20 | Surface size (use equal values for seamless tiling) |
| `--width-segments` / `--length-segments` | 128 | Grid segments per axis |
| `--wind-speed` | 5 | Wind speed (m/s) |
| `--wind-angle` | – | Wind direction in degrees (0 = +X, 90 = +Y); overrides `--wind-dir-x/y` |
| `--wind-dir-x` / `--wind-dir-y` | 1 / 0 | Wind direction vector |
| `-a, --amplitude` | 1 | Peak-to-peak height is `2 × amplitude` |
| `-c, --choppiness` | 0.5 | Horizontal displacement factor |
| `--seed` | 42 | Random seed |
| `-t, --time` | 0 | Simulation time of a still frame (conflicts with `--anim-duration`) |
| `--anim-duration` | – | Enables animation; length in seconds |
| `--anim-start` | 0 | Simulation time of the first frame |
| `--anim-fps` | 24 | Frames per second |
| `-g, --tiles` | 1 | N×N tile grid (tiles share one mesh) |
| `-o, --output` | `ocean_surface.glb` | Output file |

## Animation format

- Base mesh: a flat grid. Each frame is one morph target (`frame_000`, …) holding position
  and normal deltas.
- One weights track with linear interpolation; keyframe *i* (time `i / fps`, both endpoints
  included, so `round(duration × fps) + 1` frames) has weight 1 on target *i*.
- All frames share one height scale, chosen so the largest per-frame peak-to-peak is
  `2 × amplitude`; heights don't pump between frames.
- Every tile is a node instancing the same mesh, each with its own weights channel driven by
  one shared sampler. Tiling adds almost nothing to file size.
- File size grows as `vertices × frames × 24 bytes`; the CLI prints an estimate first.

## Whitecaps (foam)

```bash
ocean-generator --anim-duration 4 --anim-fps 24 -c 1.5 -g 3 --foam -o ocean.glb
blender --python ocean_foam/ocean_blender.py
```

`--foam` writes next to the GLB, into `<output>_foam/`:
- `foam_NNNN.png`: 16-bit grayscale foam mask per frame (row 0 = top, no flip needed for the
  GLB's UVs), `foam.json`: manifest with SHA-256 of every image, `ocean_blender.py`: imports
  the GLB, builds a water+foam material driven by the image sequence and sets fps/frame range.
- Foam is the smallest eigenvalue of the Jacobian of the choppy displacement, soft-thresholded,
  decayed exponentially (`--foam-tau`, default 2 s) and spun up before frame 0. `--foam-coverage`
  (default 0.02) is the per-frame fraction of the surface that triggers; the threshold is
  calibrated to it. Foam stays at fixed points, so it trails behind crests.
- Morph targets cannot carry foam into Blender (its importer reads POSITION deltas only), hence
  the sidecar images. All tiles show the same foam.
- `--foam-coverage physical` uses the Monahan & O'Muircheartaigh (1980) whitecap fraction for
  `--wind-speed` (only ~0.09% at 5 m/s, ~4% at 15 m/s; sources disagree by >10x at low wind).
  `--foam-substeps N` updates foam N times per frame if the trigger looks temporally aliased.
  `--foam-noise` makes the Blender material break the mask up with world-space noise.
- Needs `--choppiness > 0` and `width == length`. At default choppiness (0.5) nothing folds, so
  foam marks the most compressed crests; use `-c 1.5` or more for real breaking.
- See `docs/whitecaps-proposal.md` and `docs/whitecaps-spikes.md`.

## Seamless tiling

The FFT field is periodic over `size = max(width, length)`. Vertices sample it with wrap-around
interpolation, so with `width == length` opposite edges are identical (positions, choppy
displacement and normals). The CLI measures this and fails if the seam error exceeds 1e-4.
A non-square mesh is only a window of the periodic field and a warning is printed when tiling.

## Library

```rust
use ocean_generator::{export::export_glb, mesh::{ocean_mesh, GridSpec}, tessendorf::*};

let size = 20.0;
let params = OceanParams { resolution: 128, size, ..Default::default() };
let field = compute_wave_field(&params, 0.0);
let grid = GridSpec::new(size, size, 128, 128)?;
let mesh = ocean_mesh(&grid, size, &field);
export_glb(&mesh, None, &grid, 3, "ocean.glb".as_ref())?;
```

For animation use `frame_times` + `compute_wave_fields`, then pass `Animation { fps, frames }`
with a `flat_grid` rest pose to `export_glb`. See `examples/`.

## Verification

`cargo test` covers the unit/property tests. `tests/ref/run.sh` additionally compares the
Jacobian with an independent numpy implementation (agreement ~1e-7 on a strongly folding field).

## Python animation script

`generate_ocean_animation.py` renders per-time GLBs to a GIF (`pip install -r requirements.txt`
in a venv).
