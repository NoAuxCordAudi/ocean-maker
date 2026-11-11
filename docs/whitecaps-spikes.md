# Whitecap spikes (Phase 0) — results

Throwaway experiments run on real crate output (the spike code was not kept; method below).

## 1. Sign of the choppy displacement — WRONG in the current mesh
**Method:** periodic central differences of `displacement_x/y` on the default field; Pearson
correlation of `height` with `div D = dDx/dx + dDy/dy` over the whole grid.

| N | L (m) | wind | corr(h, div D) |
|---|---|---|---|
| 128 | 20 | 5 | **+0.81** |
| 128 | 20 | 10 | +0.85 |
| 128 | 100 | 10 | +0.71 |

Crests must *compress* (`div D < 0` at crests; deep-water orbital motion at a crest goes in the
propagation direction, `D = -k̂·a·sin(kx-ωt)` for `h = a·cos(kx-ωt)`). We measure the opposite:
crests **spread** and troughs sharpen. Analytically: with `rustfft`'s `e^{+ik·x}` inverse
transform, `D~ = -i·k̂·h~` gives `D = +k̂·a·sin θ` for `h = a·cos θ`; crest sharpening needs `+i`.

**Decision:** flip the sign (`D~ = +i·k̂·h~`) in the mesh displacement *and* use the same sign in
the Jacobian. This changes existing geometry (choppiness now sharpens crests, as intended) and is
called out in the commit/changelog.

## 2. Distribution of `lambda_min` (min eigenvalue of J) at defaults
Finite-difference estimate (attenuates short waves, so a lower bound on true folding), 128², t=0,
amplitude 1. `det J < 0` is the fraction of grid points where the mesh itself folds.

| L | wind | choppiness | p1 | p5 | p50 | min | det J<0 |
|---|---|---|---|---|---|---|---|
| 20 | 5 | 0.5 (default) | 0.69 | 0.77 | 0.94 | 0.61 | 0% |
| 20 | 5 | 1.0 | 0.38 | 0.55 | 0.88 | 0.21 | 0% |
| 20 | 5 | 1.5 | 0.07 | 0.32 | 0.82 | -0.18 | 0.65% |
| 20 | 5 | 2.0 | -0.24 | 0.09 | 0.76 | -0.58 | 3.6% |
| 20 | 10 | 0.5 | 0.75 | 0.81 | 0.94 | 0.68 | 0% |
| 20 | 10 | 2.0 | -0.02 | 0.24 | 0.78 | -0.29 | 1.1% |
| 100 | 10 | 2.0 | 0.67 | 0.76 | 0.94 | 0.56 | 0% |

**Findings**
- At default settings (c = 0.5) **nothing folds**: `lambda_min` never drops below ~0.6, so a
  physical "J < 0" trigger gives zero foam. Foam must be driven by a *calibrated threshold*
  (quantile) — i.e. the artistic mode in the proposal — not by `J < 0`.
- Real folding appears at c ≳ 1.5 for the 20 m patch.
- Larger patches (100 m) are much less choppy at the same `c` (displacement relative to
  wavelength is smaller), so threshold/coverage must be calibrated per configuration.

## 3. Blender image-sequence round trip — NOT RUN
Blender is not installed on this machine. Rules to confirm when it is (documented in the
generated script's header and tested on the Rust side only):
- glTF `v = 0` is the *top* of the image; Blender's importer flips V on import, and a PNG's
  first row is the image top, so PNG row `r` ↔ glTF `v = (r + 0.5)/M` with **no flip**.
- Texel `i` has its centre at `u = (i + 0.5)/M`, so the foam field is sampled at
  `x = (i + 0.5)/M · L` (half-texel phase) to line up with mesh vertices that sit at `u = i/N`.
- Image Sequence: `frame_start = 1`, `frame_duration = n_frames`, Extension = Repeat, Non-Color.

## 4. UV convention
Mesh UVs are `ix/N`, `iy/N` (inclusive, so the last row/column duplicates the first and wraps).
With Repeat extension and a periodic texture this is consistent; no change needed.

## 5. mesh-tools 0.22 extras / textures / extensions
Not investigated: only matters for the deferred in-GLB atlas.
