# P00 WP7 Remote-Worker Assessment and Dormant Contract

**Date:** 2026-08-12  
**Status:** conditional work package not activated; no remote service, account, credential, or data transfer was requested or performed.

## Decision basis

WP7 exists only if the exact mandatory checkpoint cannot execute correctly and reproducibly on the
local workstation. That condition is not met:

| Local evidence | Finding | Reference |
| --- | --- | --- |
| Exact source/checkpoint identity | Official DA3 source commit and `DA3NESTED-GIANT-LARGE-1.1` revision/hash are pinned and loaded locally. | `DA3_RUNTIME_LOCK.md` |
| Controlled execution | Native Windows completed finite one-, two-, and three-view outputs at 252 and 504 process resolutions. | `DA3_NATIVE_RUNTIME_COMPARISON.md`, `WP5_MEASUREMENT_RESULT.md` |
| Repeatable local baseline | Two fresh 252-pixel processes, and one persistent-worker first/warm run, produced bitwise-identical depth, confidence, intrinsics, and extrinsics for every controlled view count. | `WP5_MEASUREMENT_RESULT.md`, `WP6_NATIVE_OPTIMIZATION_RESULT.md` |
| Failure behavior | Missing weights and malformed input reject structurally; no smaller checkpoint or computation-changing compatibility patch was used. | `WP5_MEASUREMENT_RESULT.md` |

Native 504-pixel multi-view depth (and two-view extrinsics) remain numerically variable. This is a
retained P00 limitation, not a remote-worker trigger: no evidence shows that moving the identical
source/checkpoint to an unselected hardware/driver stack would correct it. No tolerance or
accepted-geometry claim has been made.

## Activation criteria

Return this assessment to the control tower before any remote action if any of the following occur:

1. The local runtime can no longer load the exact checkpoint or complete a finite controlled
   one-, two-, or three-view result in its measured envelope.
2. The 252-pixel local repeatability baseline fails under the recorded source-preserving policy.
3. A bounded P00 requirement cannot fit the documented local resource envelope, rather than merely
   exceeding an untested high-resolution stress boundary.
4. The control tower decides that testing a separately pinned hardware/driver stack is necessary
   to investigate the unresolved 504-pixel multi-view boundary.

Any activation requires control-tower authorization before provisioning a paid/external service,
creating an account, requesting credentials, or transferring any input. A remote result never
automatically replaces the selected local runtime or makes geometry accepted.

## Dormant provenance-equivalent contract

This contract is specified for a later authorized activation; it has **not** been remotely
validated. The worker must receive a content-addressed job bundle containing only approved inputs
and no credentials.

### Required request manifest

| Field | Required value |
| --- | --- |
| Schema and job identity | Versioned schema, immutable job ID, submitting code revision, submission time, and explicit retry/restart identity. |
| Source identity | DA3 source commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` and a clean source-tree check. |
| Checkpoint identity | `DA3NESTED-GIANT-LARGE-1.1`, revision `b2359bdf726fb44ef62acca04d629dcf158053e7`, safetensors SHA-256 `8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c`. |
| Code and dependency identity | Immutable repository revision plus the native DA3 core-lock identity from `DA3_RUNTIME_LOCK.md`; no source patch or unpinned substitution. |
| Inputs | Ordered one-, two-, or three-view input list, byte SHA-256 per item, MIME/type, and the requested `process_res`. Inputs must remain raw, separately retained, and access-controlled. |
| DA3 invocation | Public `DepthAnything3.inference` call with `process_res_method="upper_bound_resize"`, no unsupported batch argument, `export_dir=None`, no `infer_gs`, and no output-rendering side effect. |
| Runtime policy | The D015 deterministic controls, source-selected precision/autocast policy, upstream SwiGLU fallback, and explicit declaration that xFormers, compilation, quantization, checkpoint substitution, and vendor-source patches are disabled. |
| Output destination | A unique result location that cannot overwrite raw inputs or an earlier job result. |

### Required response manifest and artifacts

The remote response must return or make available, under the same job ID:

- source, checkpoint, code, dependency, input, and runtime-control identities actually observed;
- worker OS, Python, Torch/CUDA/driver versions, GPU name/capability, CUDA availability, precision
  selection, start/end/model-completion times, and peak system/GPU memory measurements;
- per-case raw `depth`, `confidence`, `extrinsics`, and `intrinsics` arrays, with shape, dtype,
  finite-value check, byte SHA-256, and a content hash of the full result manifest;
- explicit status, typed error and traceback/reference for failures; a failed job must not emit an
  apparently successful geometry result; and
- a retention location and access policy consistent with D012/D009. The remote worker may not
  retain inputs or outputs beyond the authorized job policy.

### Validation procedure after authorized activation

1. Verify source cleanliness, checkpoint content hash, dependency/code identity, and all input
   hashes before accepting any remote inference.
2. Run the same controlled one-, two-, and three-view inputs at 252 and 504 pixels in two fresh
   worker processes, preserving all raw results and manifests.
3. Require structural validity and finiteness in every raw output. Compare repetitions field by
   field; report exact differences rather than inventing a tolerance.
4. Exercise missing-checkpoint, malformed-input, and interrupted-worker failures; ensure typed
   failures and no false result artifact.
5. Compare the remote result envelope against the local P00 evidence. Record differences as
   hardware/runtime observations, not corrections or accepted geometry.
6. Submit the complete evidence to the control tower for a separate runtime decision. No payment,
   credentials, data transfer expansion, or canonical-runtime change is authorized by this document.

## Conclusion

The remote-worker route remains dormant because native Windows already provides finite,
provenance-bound exact-checkpoint execution and a 252-pixel repeatable baseline. WP7 adds no
external dependency or claim that remote hardware would resolve the 504-pixel multi-view boundary.
WP8 supporting-stack smoke tests remain the next P00 implementation work.

