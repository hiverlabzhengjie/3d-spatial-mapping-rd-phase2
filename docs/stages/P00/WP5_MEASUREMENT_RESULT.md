# P00 WP5 Native DA3 Measurement Result

**Date:** 2026-08-12  
**Selected runtime:** `native-windows-py311` under D014  
**Result:** measurement complete; the multi-view 504-pixel repeatability boundary remains open.

## Provenance and method

The measurement matrix used the exact source commit, checkpoint revision, checkpoint SHA-256, and
native checkpoint-core lock in `DA3_RUNTIME_LOCK.md`. Worker runs bind themselves to code revision
`ae3f14b9330f402bb60ccef99638d9b67040f163`, dependency-lock SHA-256
`6f1a5b4e0a8e6096ae0df61295fa218cf486905924b9146f5f5ed08789978b4f`, and input-manifest
SHA-256 `5e4586538a09da6e233ac03a47772ab3da005fc25c88e2b45422fc6eae62284d`.

For each resolution, the exact checkpoint ran in two fresh Python processes. Each process loaded
the checkpoint once and called DA3's public API for the fixed one-, two-, and three-view upstream
fixtures. The three-view fixture has an intentionally different third image shape, so DA3
centre-cropped it to `(98, 252)` or `(182, 504)`. It is a controlled execution test, not
same-scene geometry evidence.

The following runtime controls were set before Torch import and recorded in every worker manifest:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic=True`
- `torch.backends.cudnn.benchmark=False`

These are reproducibility controls only. The source, checkpoint, inputs, public API call, learned
parameters, and model precision policy were not changed. DA3's source-selected SwiGLU fallback
and unused optional `gsplat` renderer remain as recorded in the model/runtime lock.

## Validity, timing, and resource measurements

All 12 normal inferences produced finite `depth`, `confidence`, `extrinsics`, and `intrinsics`
arrays with the expected per-view shapes. DA3 exposes no `batch_size` parameter on its public
inference API; each result is one joint call for the stated number of views.

| Resolution | Case | Wall time range | Model-forward range | Peak allocated / reserved VRAM | Process RSS range | Repeat result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 252 | One view | 0.758-0.880 s | 0.714-0.834 s | 9.131 / 9.326 GiB | 0.914-0.923 GiB | all fields bitwise identical |
| 252 | Two views | 0.433-0.441 s | 0.407-0.415 s | 9.176 / 9.320 GiB | 0.928-0.939 GiB | all fields bitwise identical |
| 252 | Three views | 0.523-0.543 s | 0.494-0.515 s | 9.177 / 9.377 GiB | 0.968-0.977 GiB | all fields bitwise identical |
| 504 | One view | 1.110-1.155 s | 1.056-1.103 s | 9.246 / 9.420 GiB | 0.921-0.928 GiB | all fields bitwise identical |
| 504 | Two views | 1.291-1.294 s | 1.255-1.258 s | 9.395 / 9.812 GiB | 0.996-1.001 GiB | confidence/intrinsics exact; depth/extrinsics drift |
| 504 | Three views | 1.381-1.389 s | 1.339-1.350 s | 9.390 / 9.783 GiB | 1.035-1.040 GiB | confidence/intrinsics/extrinsics exact; depth drift |

The host reported 31.934 GiB system RAM. Available RAM before normal model loads ranged from
23.360 to 23.380 GiB and remained at least 22.819 GiB after the measured cases. Exact checkpoint
CPU load time ranged from 8.698 to 8.717 s; GPU transfer ranged from 1.650 to 1.750 s.

The outer task-runner's 124-second window interrupted the matrix orchestrator after it had started
the final failure probe. All four normal worker manifests and the missing-checkpoint rejection had
already completed; the malformed-input worker completed and was found by the resume step. The
individual model workers were not reported as interrupted. Because that outer interruption
prevented the first orchestrator from assembling its `nvidia-smi` snapshots, this matrix is not an
absolute low-idle GPU performance baseline. Its PyTorch memory and process-RAM measurements remain
valid; later benchmarks must retain per-run external GPU snapshots.

## Repeatability evidence

At 252 pixels, every field in every one/two/three-view case was bitwise identical across the two
fresh processes. At 504 pixels, one view was also bitwise identical.

At 504 pixels, the two-view result had depth differences at all 282,240 elements, with maximum
absolute delta `0.0195732` and mean absolute delta `0.00652291`; six of 24 extrinsic elements also
differed, with maximum delta `0.0000282526`. At 504 pixels, three-view depth differed at 45,947 of
275,184 elements, with maximum/mean absolute deltas `0.0224171` / `0.00374294`; extrinsics were
bitwise identical. Confidence and intrinsics were bitwise identical in both multi-view cases.

These are DA3 output-scale comparisons, not metres or acceptance tolerances. P00 does not invent a
numerical tolerance here. The result supports reproducible execution at 252 pixels and one-view
504 pixels, but it does not establish bitwise-repeatable multi-view accepted geometry at 504 pixels.
Any later tolerance must be derived from the strongest-camera pilot and independent physical
validation under D008 and D010.

## Retained failure behavior

| Probe | Observed behavior | Evidence |
| --- | --- | --- |
| Missing checkpoint | Structured `FileNotFoundError` before model load, naming the missing `model.safetensors` path | `failure_missing_checkpoint.json` |
| Malformed image payload | Structured `PIL.UnidentifiedImageError` from DA3 input processing after a local checkpoint load; no result arrays written | `failure_malformed_input.json` |
| Unsupported `batch_size` keyword | Pinned public API rejected it with `TypeError`; retained during WP4 | `DA3_NATIVE_RUNTIME_COMPARISON.md` |
| Outer orchestration timeout | Task runner ended the parent at 124 s; individual completed worker artifacts were resumed without rerun | Parent interruption retained in this record and `LOG.md` |

No high-resolution OOM probe was run. The largest measured case reserved 9.812 GiB of the reported
11 GiB VRAM, leaving limited headroom on an interactive workstation. An OOM probe would add little
accepted-geometry evidence and may be considered later only as an explicitly bounded stress test.

## Artifact inventory and conclusion

The full summary, four normal worker manifests, two failure manifests, 12 raw compressed prediction
files, worker logs, and the controlled malformed input are retained at:

`X:\3D-Spatial-Mapping-Phase2\runs\p00-wp5-native-matrix-20260812`

WP5 verifies output validity, the tested resolution/view-count resource envelope, clean-process
repeatability characteristics, and two input failure paths for the selected native runtime. It does
not close the multi-view 504-pixel repeatability gate or authorize accepted geometry. WP6 should
evaluate only source-preserving runtime controls, and WP8 must still close the supporting-stack
smoke tests.

