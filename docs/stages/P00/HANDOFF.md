# P00 Stage Handoff - Project foundation and compute feasibility

**Dedicated task:** Phase 2 - P00 - Project foundation and compute feasibility
**Submitted:** 2026-08-12
**Recommended outcome:** partial

## Outcome

P00 established a safe artifact boundary, compared pinned native-Windows and WSL2 candidates,
and proved that the exact mandatory `DA3NESTED-GIANT-LARGE-1.1` checkpoint runs locally without a
source patch, smaller model, or computation-altering workaround. Native Windows is the canonical
runtime recommendation. The selected supporting stack and a limited Phase 1 reuse port are tested.

The stage cannot recommend full acceptance because controlled 504-pixel multi-view runs remain
numerically variable under the measured deterministic controls. P00 retains a bitwise-repeatable
252-pixel one/two/three-view baseline and a bitwise-repeatable 504-pixel one-view result, but it
does not claim accepted multi-view geometry or invent an output tolerance.

## Implemented capabilities

- Isolated, validated `X:\3D-Spatial-Mapping-Phase2` storage for model weights, captures, cache,
  runs and exports, while source, tests and lightweight records remain on `C:`; credentials remain
  outside Git.
- Comparable CPython 3.11.4 native-Windows and WSL2 candidates with common DA3 smoke definition,
  pinned inputs, manifests, validity checks and retained raw outputs.
- Native canonical runtime under D014, with a D015 bounded persistent-model lifecycle that preserves
  DA3's source-selected FP16/SwiGLU fallback and deterministic controls. xFormers, compilation,
  manual precision changes and related unvalidated accelerators remain prohibited.
- Native supporting stack under D017: GeoCalib, OpenCV, SciPy, Open3D, Rerun, PyAV/FFmpeg, HEIF,
  pre-commit, FastAPI and Uvicorn completed bounded representative operations.
- A dormant D016 remote-worker contract, not provisioned because the exact checkpoint remains
  locally viable.
- A tested, dependency-free `TimestampTransform` contract adapted from the pinned Phase 1 source.

## Inputs and provenance

- Exact DA3 source: `https://github.com/ByteDance-Seed/Depth-Anything-3` commit
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- Exact mandatory checkpoint: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, revision
  `b2359bdf726fb44ef62acca04d629dcf158053e7`, `model.safetensors` SHA-256
  `8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c`.
- Canonical interpreter: native CPython 3.11.4 at
  `X:\3D-Spatial-Mapping-Phase2\cache\runtime_envs\native-windows-cpython311\Scripts\python.exe`
  with `torch==2.0.1+cu118`, `torchvision==0.15.2+cu118` and `numpy==1.26.4`.
- Supporting stack: GeoCalib commit `97b8968e7798a66bf04fcf791fb535624241bda7`, official
  pinhole weight SHA-256 `86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac`,
  and `requirements/p00-native-supporting-direct.lock`.
- Phase 1 audit source: clean commit `e9e038c498c17d83bcfa65a71e1c355e3f4aa8d7` from the
  documented project repository; no sensitive artifact or runtime configuration was copied.

The complete model/runtime identity and reproduction entry point are in `DA3_RUNTIME_LOCK.md`.

## Evidence and verification

- Both native Windows and WSL2 completed controlled finite one-, two-, and three-view inference at
  DA3's 504-pixel default. Native loaded about 5.2x faster and inferred about 2.1-2.7x faster with
  essentially the same tracked peak VRAM; see `DA3_NATIVE_RUNTIME_COMPARISON.md`.
- WP5 completed two clean processes at 252 and 504 pixels for all view counts, recording raw
  predictions, hashes, timing, system RAM, allocated/reserved VRAM and expected failure behavior.
  Peak reserved VRAM was 9.812 GiB in the measured 504-pixel two-view case; see
  `WP5_MEASUREMENT_RESULT.md` and its D: run store.
- WP6's one-loaded-model probe was bitwise exact across first/warm 252-pixel one/two/three-view
  calls. Its 504-pixel multi-view non-exact results are preserved rather than masked; see
  `WP6_NATIVE_OPTIMIZATION_RESULT.md`.
- WP8 passed bounded GeoCalib, OpenCV, SciPy, Open3D, Rerun, PyAV/FFmpeg and web-stack operations,
  then reran the exact DA3 252-pixel one/two/three-view regression successfully; see
  `WP8_SUPPORTING_STACK_RESULT.md`.
- WP9 passed `43` tests, `ruff check src tests`, and `mypy src tests` across 14 source files; see
  `WP9_PHASE1_REUSE_ASSESSMENT.md`.

`EVIDENCE.md` indexes the full run manifests, hashes, raw-output stores, test results and retained
negative findings.

## Decisions and deviations

- D012 separates C: tracked source from the verified D: artifact root.
- D013 preserves comparable CPython 3.11.4 candidates; D014 selects native Windows based on measured
  performance, while retaining WSL2 evidence.
- D015 permits only a provenance-bound persistent exact-model worker; D016 keeps remote execution
  dormant; D017 pins the supporting stack without changing the DA3 core.
- P00 tested only GeoCalib's official pinhole weights. Its `distorted` weights and radial/divisional
  model candidates are explicitly deferred to P04 and must be evaluated against clean frames,
  stream-profile provenance, shared-intrinsic stability and held-out evidence.

## Failures, limitations and residual risks

- At 504 pixels, two-view depth differed at all 282,240 elements across clean processes (maximum
  delta `0.0195732`, mean `0.00652291` in DA3 output scale), with a small two-view extrinsic drift.
  Three-view depth also differed (45,947 of 275,184 elements; maximum `0.0224171`, mean
  `0.00374294`). These are not metre-scale or acceptance tolerances.
- The persistent worker did not remove this 504-pixel multi-view variation. No 504-pixel multi-view
  result is accepted geometry, and remote hardware is not an approved workaround.
- High-resolution/OOM limits remain unmeasured on the interactive 11 GiB host.
- `xformers` is deliberately absent; `pip check` reports that expected upstream declaration only.
  GeoCalib's current Kornia 0.8.3 and current Rerun 0.36 were rejected as incompatible with the
  DA3 core; their retained failures are in WP8 evidence.
- The supporting stack smoke makes no calibration-accuracy claim. Real camera frames, RTSP profiles,
  floor-plan control, camera poses and physical geometry tolerances are future-stage evidence.

## Repository state

Relevant P00 commits are `f6ebcbc` (cross-runtime exact DA3 smoke), `0e50760` (measurements),
`626d4d9` (native optimization evidence), `01666e7` (remote contingency), `30c73b0` (supporting
stack closure), and `7277096` (Phase 1 reuse assessment). The branch is local `main`; no private
Git remote has been supplied or verified. This handoff is submitted in the final P00 handoff commit.

## Downstream contract

If partially accepted, P01-P04 may use the verified artifact layout, native runtime recommendation,
pinned dependency/model identities, D015 worker policy, supporting-stack selections and the P00
timestamp-transform contract. They must retain per-run provenance and raw outputs.

No downstream stage may treat P00's synthetic fixtures as office geometry, treat 252-pixel exact
execution as proof of real-scene accuracy, claim a 504-pixel multi-view tolerance, enable a rejected
accelerator, substitute the checkpoint, or activate a remote worker without control-tower authority.
P04 must validate GeoCalib distortion candidates; P06 must derive geometry-use criteria from later
physical and held-out evidence.

## Required control-tower action

Review this handoff for **partial acceptance**. Confirm whether the verified native foundation can
be used by P01-P04 under the stated restrictions, while retaining 504-pixel multi-view DA3 outputs
as diagnostic/experimental only until later physical validation defines and reviews an appropriate
criterion. Return a bounded revision brief only if additional P00 evidence is required before that
restricted continuation.

