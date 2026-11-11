# Whitecaps for the Ocean Generator: Final Proposal

Branch: `whitecaps-research`

## 0. Plan review (what I corrected)
- **All 3 plans agree** on the core design. Foam is computed in Rust from the spectral Jacobian and delivered as a sidecar PNG sequence plus a Blender setup script. That matches the dossier.
- **Plan 2 (physical default, tau 3.5 s, 60 Hz internal step, 2x oversample) rests on weak ground.**
  - A physical MOM80 default gives about 0.09% coverage at 5 m/s. The dossier says that reads as no foam, and the parameterisations disagree by more than 10x at the 5 m/s knee.
  - tau of 3.5 s looks static on a 4 s clip. The dossier says avoid more than about 3 s.
  - A 16 s spin-up at 60 Hz is extra complexity for a first version.
- **Plan 1 and Plan 3 defaults are artistic.** Plan 1 uses 2% trigger coverage and tau 1.5 s. Plan 3 uses 3% steady-state coverage and tau 2 s. Both are consistent with the dossier. I take the artistic default and keep physical coverage as an option.
- **Unverified assumptions in the plans:**
  - Smooth import recomputes normals correctly. Derived from source, never observed.
  - Image Sequence playback, frame offset and V orientation in Blender. Source only.
  - Plan 1 hedges on whether the sign fix is needed. Plan 3 and Plan 2 assume the flip is needed.
  - Plan 3's half-texel phase factor is a good idea but untested.
  - Plan 3's claim that the PNG needs no flip is also untested.
  - Blender version: unknown.
- **Plan claims that touch verifier corrections (all handled):**
  - Zero-padding needs no `(M/N)^2` scale with unnormalised rustfft.
  - Finite differences give 0.64 at half-Nyquist and 0 at Nyquist. They are a test oracle only.
  - MM80 has no 3.7 m/s lower bound. That bound belongs to Callaghan08.
  - MM80 at 15 m/s is 3.9%, not 2.6%.
  - Tessendorf's "half-life" is really an e-folding time.
  - Blender's live modifier can show baked accumulated foam.
- **Plan 3's formula text has an inconsistent sign** between "literal -i gives positive multipliers" and "after flip sigma is -1". I use one derivation below, with a single sign constant that the self-test decides.

## 1) Recommendation
- **Compute foam in Rust.** Use the spectral Jacobian minimum eigenvalue (`lambda_min`), a soft trigger, exp decay, and deterministic spin-up.
- **Ship foam as a PNG sequence on the existing 0..1 UVs, plus `foam.json` and a generated Blender setup script.** Morph targets cannot carry foam, because Blender reads POSITION only.
- **Calibrate the threshold to a target coverage.** Default is artistic (2% per-frame trigger). Physical MOM80 is an opt-in mode. Fail fast if foam is unreachable.
- **Run spikes first.** The sign of D, the `lambda_min` histogram at defaults, and a Blender image-sequence round trip can each change the plan.
- **The GLB stays foam-free.** An in-GLB atlas (KHR_animation_pointer) is deferred until a spike proves it.

## 2) How whitecaps work here

**Conventions.**
- `h~(k,t)` is the existing time-evolved spectrum.
- `s` is the shared animation scale.
- `c` is choppiness (0.5).
- `k_hat = k/|k|`.
- `SIGN` in {+1,-1} is one constant used by both mesh and Jacobian.
- Mesh displacement spectrum: `D~_a = SIGN * i * s*c * k_hat_a * h~`, applied as x+Dx, y+Dy.

**Sign (dossier-verified issue).**
- Under rustfft's e^{+ik.x}, crest sharpening needs `+i`. The literal `-i` spreads crests and folds troughs.
- Whether our index mapping flips this is unknown. The self-test decides.

**Jacobian (3 extra real IFFTs per frame).**
- `D_ab(x) = IFFT[ -SIGN * s*c * k_a*k_b/|k| * h~ ]` for a,b in {x,y}.
- The multiplier is 0 at k=0.
- `Dxy = Dyx`, because D is a gradient field.
- Nyquist row/column: zero the multipliers. Take the real part and assert `max|imag| < 1e-4 * max|real|`.
- `Jxx = 1+Dxx`, `Jyy = 1+Dyy`, `Jxy = Dxy`.
- `lambda_min = 0.5(Jxx+Jyy) - 0.5*sqrt((Jxx-Jyy)^2 + 4*Jxy^2)`, with the sqrt in f64.
- No eigenvectors, so there is no `Jxy=0` hazard.
- Keep `det J = Jxx*Jyy - Jxy^2` as a mesh-fold diagnostic.
- Single-mode check: `h = A cos(kx)` gives `J = 1 - s*c*A*k*cos(kx)` with the crest-sharpening sign.

**Trigger (soft).**
- `g = saturate((lambda_T - lambda_min)/w)`, with `w = 0.15`.
- The ramp is a demo heuristic, not Tessendorf. Document that.
- Hard threshold is `w -> 0`.

**Temporal update (pointwise, no advection).**
- `alpha = exp(-dt/tau)`.
- `F_{n+1} = max(F_n*alpha, g_{n+1})`, with `F_0 = 0`.
- `tau` is an e-folding time. Half-life is `tau*ln2`.
- Default `tau = 2.0 s`. Crest uses 1.25 s. Observed decay is 1.4-4.8 s. Tessendorf's 8 s looks static on a 4 s clip.

**Spin-up.**
- Run `n_warm = ceil(ln(0.01)/ln(alpha))` frames before `t_start`, about 4.6 tau.
- Use negative or earlier times, with the same `s`.
- Discard these frames. Cost is IFFTs only.
- The shared `s` is fixed from the clip frames first (pass 1). The spin-up frames then reuse it (pass 2).

**Threshold calibration.**
- Default `--foam-coverage 0.02` is a per-frame trigger fraction.
- Pool `lambda_min` over spin-up and clip frames in fixed order. Take the empirical quantile as `lambda_T`.
- Steady-state coverage ends higher than 0.02 because of decay: `<W> = Theta/(1 - alpha(1-Theta))`.
- Tessendorf's CDF form is `CDF(lambda_T) = W(1-alpha)/(1-alpha*W)`. Use it as a unit-test cross-check.
- Why artistic: MOM80 `W = 3.84e-6 * U^3.41` (a fraction) gives 0.093% at 5 m/s.
- Fail fast if:
  - `lambda_T >= 0.98`. Message: raise `--choppiness` or amplitude.
  - Realised mean F is 0.
  - Choppiness is 0.
- Tessendorf's example thresholds (0.71, 0.81) came from 400-4000 m patches and do not transfer.

**Output resolution.**
- v1 default is oversample 2x (256x256). Compute the three Jacobian fields on a zero-padded grid.
- Apply the nonlinearity at the fine resolution.
- No `(M/N)^2` scale. Include the half-texel phase so texel (i,j) sits at `x = (i+0.5)/M * L`. Verify with a test pattern.
- 16-bit grayscale PNG, Non-Color in Blender.
- Foam sits on undisplaced grid points, so it is Lagrangian and periodic. It rides the orbital motion, but crests move by phase and leave trails. Accepted trade-off.

## 3) How foam gets into Blender

**Chosen: sidecar PNG sequence + generated bpy script.**
- The script runs `blender --python out/ocean_blender.py`.
- It reads the manifest and verifies hashes and frame count. Fail fast on mismatch.
- It sets scene fps to the manifest fps **before** import, plus frame range.
- It imports with `bpy.ops.import_scene.gltf(import_shading='SMOOTH')`.
- It builds one material: water Principled (blue, IOR 1.33) mixed with foam Principled (base 0.9, roughness 0.7, specular IOR level 0.2, subsurface 0).
- The Image Texture node uses Source=Sequence, `frame_duration = n`, `frame_start = 1`, auto-refresh, Extension=Repeat, Non-Color, Linear. UVs come from the UV Map node. The material is assigned to all tiles.
- It uses no frame handlers, so it works in headless and farm renders.
- The GLB geometry path is unchanged.

**Evidence for the constraint.**
- Importer `mesh.py` reads only `prim.targets[sk]['POSITION']`. COLOR, TEXCOORD, NORMAL and custom morph deltas are silently ignored (read in source by three researchers).
- mesh-tools 0.22 `MorphTarget` has only positions, normals and tangents.
- Smooth shading: with NORMALS, the flat base grid is detected as flat and base normals are frozen. SMOOTH recomputes from shape keys. This is derived from source, not yet observed.

**Rejected alternatives.**
- **COLOR_n, TEXCOORD or custom morphs.** Importer drops them. Writer cannot emit them.
- **NORMAL/TANGENT abuse.** Ignored by the importer.
- **Static COLOR_0.** Not animated, 8-bit BYTE_COLOR, sRGB-encoded in Blender, and it tints the water in other viewers.
- **frame_change handler writing a foam attribute.** Render-thread and depsgraph risk, unverified.
- **USD or Alembic.** No verified Rust writer with primvar time samples. Blender's animated-primvar support is limited and not confirmed for per-frame colour. `openusd` has writers but no documented primvar or time-sample authoring.
- **Geometry Nodes proxy.** Stateless, no decay, differs from the Rust reference. Optional preview and cross-check only. Use XY-projected area or a grid-neighbour Jacobian, not 3D face area.
- **In-GLB atlas** (`KHR_texture_transform` + `KHR_animation_pointer`, STEP). Deferred:
  - Unproven STEP behaviour.
  - Offset, rotation and scale keys must share identical key times, or the importer skips offset conversion.
  - Needs Blender 4.2 or later.
  - Pointer import is marked experimental.
  - Needs atlas gutters.
  - mesh-tools support is unverified, so GLB JSON post-processing may be needed.

**Known visual costs.**
- Every tile shows identical foam. Mitigation is world-space noise breakup in the shader, with a baked periodic detail PNG.
- 96 weight-animated nodes each get their own mesh with all shape keys. That is a Blender memory cost, not foam-specific.

## 4) Phased plan

**Phase 0: Spikes (about 1 day, throwaway).**
- Dump the `lambda_min` histogram at defaults (N=128, 20 m, 5 m/s, c=0.5) and at c in {0.5,1,1.5,2}.
- Run the crest/trough J sign test on real crate output.
- Do a hand-built 3-frame Blender round trip with an asymmetric pattern. Check frame offset, V flip and half-texel alignment. Record the Blender version.
- Check the mesh UV convention (`i/N` vs `i/(N-1)`, seam duplicates).
- Exit:
  - Sign result recorded.
  - `lambda_min` percentiles known, and a decision on whether default choppiness must rise for foam to appear.
  - Blender sequence mapping rules written down.
  - UV convention confirmed.

**Phase 1: Jacobian (src/tessendorf.rs).**
- Add the `SIGN` constant, `JacobianFields`, `lambda_min`, the imag-residual assert, and the sign self-test.
- Flip D if the self-test requires it. That is a geometry change: call it out in the changelog and regenerate goldens deliberately.
- Exit:
  - Single-mode analytic test passes. `c=0` gives `J=1` and `lambda_min=1`.
  - Spectral derivatives match periodic finite differences on band-limited low-k fields. `Dxy` matches FD `dDy/dx`.
  - Zero-padded values equal coarse values at coincident samples, with no 4x inflation.
  - Self-test passes on the default clip.

**Phase 2: Foam model (src/foam.rs).**
- Add `FoamParams`, `step`, spin-up, calibration (quantile plus bisection on realised mean F), and typed errors.
- Exit:
  - Decay-only equals `F0*exp(-t/tau)` to 1e-6. Two steps of dt equal one of 2dt for decay.
  - Extra spin-up changes frame 0 by less than eps.
  - Realised trigger fraction is within 20% of the target. Calibration errors correctly.
  - Same seed and parameters give byte-identical output.

**Phase 3: Export + Blender script.**
- PNG writer (deterministic `png` crate settings), `foam.json` with sha256, generated script, README.
- Exit:
  - Headless Blender round trip: frame K sampled through UVs matches `foam_K.png` within filtering tolerance.
  - Foam sits on crests and fades behind them, not in troughs.
  - Wrap-around seam test and a 2x2 tile render show no seams.
  - Shape-key count equals frame count, enforced by the script.

**Phase 4: Look-dev and validation (each item optional).**
- Detail noise, bump and roughness from foam.
- Blender Ocean modifier structural comparison.
- Wind sweep.
- 2-IFFT packing.
- `--foam-substeps 2` if temporal aliasing shows at 24 fps.
- Physical coverage mode.
- Self-contained GLB atlas, only if a Blender spike proves it.

## 5) CLI/API changes
All opt-in. Without `--foam`, the GLB is byte-identical to today, except the deliberate sign-flip regeneration if needed.
- `--foam`: enable, and write the PNG sequence, manifest and Blender script.
- `--foam-coverage <frac|physical>`: default 0.02 artistic. `physical` uses MOM80 from `--wind`, and the help text states the source.
- `--foam-threshold <f32>`: explicit `lambda_T`, bypasses calibration.
- `--foam-softness <w>`: default 0.15.
- `--foam-tau <s>`: e-folding, default 2.0.
- `--foam-spinup <s>`: default auto, about 4.6 tau. A shorter value warns with the residual bound.
- `--foam-resolution <M>`: default 256, even and at least N.
- `--foam-substeps <K>`: default 1.
- `--foam-dir <path>`, `--no-blender-script`, `--foam-debug-dir <dir>`.
- Rust API:
  - `tessendorf::{SIGN, JacobianFields, jacobian_fields, lambda_min, verify_fold_sign}`.
  - `foam::{FoamParams, step, simulate_foam, calibrate_threshold, mom80}`.
  - `export::write_foam_sequence`.
  - All fallible functions return typed errors. No silent clamping.
- Determinism: fixed order, single-threaded reductions, same-platform f32.

## 6) Validation plan

**Independent references.**
- **Finite-difference oracle.** Periodic central differences of the exported displacement, on band-limited low-k fields only. FD gain is `sin(k dx)/(k dx)`: 0.64 at half-Nyquist, 0 at Nyquist.
- **Mesh-fold check.** Where exported XY triangle areas flip sign on a high-chop field, `lambda_min` and `det J` are at or below 0.
- **Numpy reference.** `tests/ref/foam_ref.py` in a venv. Rust dumps `h~` for N=16/32. Numpy recomputes the Jacobian, `lambda_min` and the recurrence, compared to about 1e-5.
- **Blender Ocean modifier (HOT lineage).** Scripted bake with matching size, wind and chop. Compare structure only: foam vs negative `J-` correlation, crest co-location, coverage trend. Not pixel equality, because:
  - Blender's Jacobian omits scale and normalisation.
  - It uses slope -0.005.
  - It uses Y-up axes.
  - The original HOT source was not read.
- **Blender round trip.** Frame K image-sequence pixels equal `foam_K.png`. Also check lighting changes between frames, to confirm normals are recomputed.

**Invariants.**
- `c=0` gives `J=1`, `lambda_min=1`, zero foam.
- `lambda_min <= lambda_max`. `det J = lambda_min * lambda_max` to 1e-5. `Dxy = Dyx`.
- F stays in [0,1], is finite, and is non-increasing without a trigger.
- Coverage is monotone in threshold.
- Periodic wrap continuity. All tiles share one buffer. No non-wrapping filters.
- Calibration identity: a stationary Bernoulli trigger gives `<W> = p/(1-alpha(1-p))`.
- Unit-trap tests: MOM80 `3.84e-6` is a fraction, `3.84e-4` is percent. Values are 0.093% at 5 m/s, 0.99% at 10, 3.9% at 15.
- Determinism: same inputs give byte-identical PNGs and manifest hash.
- Optional: run the Khronos glTF-Validator if the GLB is modified.

## 7) Risks, trade-offs, first spikes

**Risks.**
- **Defaults may never fold.** At 5 m/s and c=0.5, `lambda_min` may stay above 0.8. The quantile mode then places foam on non-breaking crests, which is not physical. The spike histogram decides, and fail-fast guards the extreme case.
- **Sign flip alters existing geometry.** Needs a changelog entry.
- **Blender behaviours are source-derived, not run.** Frame offset, V flip, Smooth normals, Non-Color, STEP.
- **No advection.** Foam leaves trails at fixed material points.
- **Coarse foam.** Mitigated by 2x oversample and soft ramp.
- **Identical tile foam.**
- **Blender memory** for many tiles times 96 shape keys.
- **Temporal aliasing at 24 fps.**
- **Heuristic constants** (`w`, tau, Blender node values) are tuned by eye.

**Trade-offs (explicit).**
- **Sidecar plus script vs one-file GLB.** Robust, but not self-contained.
- **Spectral vs FD Jacobian.** 3 extra IFFTs, exact and periodic. FD attenuates short waves where crests break.
- **`lambda_min` vs `det J`.** `lambda_min` is more sensitive. `det J` is kept as a diagnostic.
- **`max(F*alpha, g)` vs additive accumulation.** dt-robust and testable, less build-up. Blender's `prev^2*fade` is frame-rate dependent and not copied.
- **Artistic vs physical coverage.** Visible foam, but not physically calibrated.
- **Jacobian uses scaled D.** This is consistent with the mesh, deliberately unlike Blender.
- **Deferred:** spray, oriented streaks (eigenvector), per-band energy, Dupuy-Bruneton filtering, acceleration or curvature trigger, 60 Hz internal step.

**First spikes (cheapest first).**
1. Sign test on real output.
2. `lambda_min` histogram and `det J` at c in {0.5,1,1.5,2}.
3. Blender 3-frame round trip.
4. Whether mesh-tools 0.22 can emit extras or textures (only matters for the deferred atlas).
5. Blender Ocean bake harness.

## 8) What the research could NOT establish
- Which sign the crate's FFT and index convention gives.
- The `lambda_min` distribution at defaults, and hence whether foam is reachable at 5 m/s without raising choppiness.
- Any Blender behaviour at runtime: Smooth-normal recompute, Image Sequence frame offset, V orientation, STEP on pointer animation. Blender version unknown. The importer was read on `main`.
- mesh-tools 0.22 support for extras, textures, extensions and pointer channels.
- Primary-source reads of Monahan 1980, Callaghan 2008 and 2012. Values come from secondary quotes. Salisbury 2013 vs 2014 coefficients differ.
- Original HOT source, Crest foam internals, NVIDIA WaveWorks and Houdini foam.
- Dupuy-Bruneton and Crest claims. Atlas, Sea of Thieves and War Thunder values are unverifiable.
- Whether 2x oversampling beats bilinear upsampling visually. Gibbs ringing near cusps.
- Tessendorf's 2022 slides (garbled extraction). The 2023 paper is the authority.
- Current Blender 5.x USD and Alembic animated-attribute behaviour.
- Current Blender defaults for `foam_coverage` and `foam_fade`. Read only from the v4.2.0 tag.

## 9) Sources (URLs read)
- https://jtessen.people.clemson.edu/reports/papers_files/coursenotes2004.pdf
- https://jtessen.people.clemson.edu/reports/papers_files/whitecap_fraction.pdf
- https://jtessen.people.clemson.edu/reports/papers_files/Os2022_tessendorf_reinhardt_gao.pdf
- https://raw.githubusercontent.com/blender/blender/main/source/blender/blenkernel/intern/ocean.cc
- https://raw.githubusercontent.com/blender/blender/main/source/blender/modifiers/intern/MOD_ocean.cc
- https://raw.githubusercontent.com/blender/blender/main/source/blender/makesrna/intern/rna_modifier.cc
- https://raw.githubusercontent.com/blender/blender/v4.2.0/source/blender/makesdna/DNA_modifier_defaults.h
- https://docs.blender.org/manual/en/latest/modeling/modifiers/physics/ocean.html
- https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html
- https://docs.blender.org/manual/en/latest/render/shader_nodes/textures/image.html
- https://docs.blender.org/manual/en/4.0/files/import_export/usd.html
- https://docs.blender.org/api/current/bpy.types.OceanModifier.html
- https://developer.blender.org/docs/release_notes/4.2/pipeline_assets_io/
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/mesh.py
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/animation_pointer.py
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/animation_weight.py
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/node.py
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/blender_gltf.py
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/pbrMetallicRoughness.py
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/texture.py
- https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/__init__.py
- https://github.com/KhronosGroup/glTF-Blender-IO/blob/main/docs/blender_docs/scene_gltf2.rst
- https://github.com/KhronosGroup/glTF-Blender-IO/issues/2285
- https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- https://raw.githubusercontent.com/KhronosGroup/glTF/main/extensions/2.0/Khronos/KHR_animation_pointer/README.md
- https://raw.githubusercontent.com/KhronosGroup/glTF/main/extensions/2.0/Khronos/KHR_texture_transform/README.md
- https://raw.githubusercontent.com/KhronosGroup/glTF/main/extensions/2.0/Khronos/KHR_node_visibility/README.md
- https://docs.rs/mesh-tools/0.22.0/mesh_tools/struct.MorphTarget.html
- https://docs.rs/mesh-tools/0.22.0/mesh_tools/struct.PrimitiveSpec.html
- https://docs.rs/mesh-tools/0.22.0/mesh_tools/builder/struct.GltfBuilder.html
- https://github.com/2Retr0/GodotOceanWaves
- https://raw.githubusercontent.com/wave-harmonic/crest/master/crest/Assets/Crest/Crest/Shaders/Resources/UpdateFoam.compute
- https://raw.githubusercontent.com/wave-harmonic/crest/master/crest/Assets/Crest/Crest/Scripts/LodData/Settings/SimSettingsFoam.cs
- https://luma.gl/docs/api-reference/experimental/spectral-ocean-simulation
- https://liris.cnrs.fr/Documents/Liris-5812.pdf
- https://gpuopen.com/gdc-presentations/2019/gdc-2019-agtd6-interactive-water-simulation-in-atlas.pdf
- https://developer.download.nvidia.com/assets/gameworks/downloads/regular/events/cgdc15/CGDC2015_ocean_simulation_en.pdf
- https://eprints.whiterose.ac.uk/77830/1/Salisbury_2013_JGR_jgrc20436.pdf
- https://eprints.whiterose.ac.uk/78109/7/Salisbury_2014_GRL_grl51407.pdf
- https://etheses.whiterose.ac.uk/id/eprint/7706/1/DJSALISBURY_Thesis.pdf
- https://acp.copernicus.org/articles/16/13725/2016/acp-16-13725-2016.pdf
- https://www.weather.gov/mfl/beaufort
- https://arxiv.org/html/2505.14230v1
- https://github.com/mxpv/openusd/blob/main/ROADMAP.md
- https://docs.rs/crate/ogawa-rs/latest
- https://antoniospg.github.io/UnityOcean/OceanSimulation.html
- https://history.siggraph.org/wp-content/uploads/2022/09/2018-Talks-Ang_The-Technical-Art-of-Sea-of-Thieves.pdf
- https://github.com/iodiot/webgpu-ocean
- https://crest.readthedocs.io/en/4.20/user/water-appearance.html
