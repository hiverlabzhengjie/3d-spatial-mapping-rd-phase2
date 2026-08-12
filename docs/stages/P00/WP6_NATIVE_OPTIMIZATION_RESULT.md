# P00 WP6 Native DA3 Optimization Result

**Date:** 2026-08-12  
**Selected runtime:** native Windows CPython 3.11.4 under D014  
**Result:** a bounded persistent-model worker is permitted; no computation-altering accelerator is accepted.

## Scope and provenance

WP6 tested only a lifecycle optimization: load the mandatory `DepthAnything3` model once in one
native GPU process, then serve the fixed one-, two-, and three-view calls from that unchanged
model instance. It does not implement a production job service, change learned parameters, alter
the public DA3 inference call, substitute a checkpoint, or conceal a failure.

| Item | Identity |
| --- | --- |
| Probe implementation | `scripts/run_p00_da3_wp6_persistent_probe.py`, code revision `bb816648aeb4c0c3594b4bea92d63fe1841b5284` |
| Guarded policy | `src/spatial_mapping_phase2/native_inference_policy.py` |
| DA3 source | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| Mandatory checkpoint | `DA3NESTED-GIANT-LARGE-1.1`, revision `b2359bdf726fb44ef62acca04d629dcf158053e7` |
| Safetensors SHA-256 | `8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c` |
| Native device | NVIDIA GeForce GTX 1080 Ti, compute capability `6.1` |
| Source-selected precision | FP16; `torch.cuda.is_bf16_supported()` was false |
| Inputs | Pinned upstream fixtures from WP4/WP5; the three-view case intentionally has a differently shaped third image and is not same-scene geometry evidence |

Every probe run set the following before importing Torch:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic=True`
- `torch.backends.cudnn.benchmark=False`

DA3 retained its unmodified source choice of FP16 on this GPU and its source-provided SwiGLU
fallback when xFormers is absent. The optional `gsplat` renderer was not requested.

## Allowed and rejected controls

| Control | Status | Reason |
| --- | --- | --- |
| Reuse one loaded exact model in one bounded GPU worker | permitted | Avoids repeated 10.3-11.0 s checkpoint loads without changing a model call. |
| Tested process resolutions 252 and 504 | permitted for execution evidence | The policy rejects unmeasured resolutions; this does not make either resolution accepted geometry. |
| Source-selected FP16 and upstream SwiGLU fallback | required | These are the pinned source's normal choices on the GTX 1080 Ti. |
| cuBLAS/cuDNN deterministic controls above | required | Measured reproducibility controls; they are not a numerical-correctness patch. |
| xFormers, `torch.compile`, manual precision override, TF32, quantization, pruning, checkpoint substitution, or vendor-source modification | rejected | Unvalidated implementation or learned-computation change. |
| `infer_gs` or export rendering | disabled for this probe | Outside the inference optimization measurement; `gsplat` is optional and absent. |

An operational worker using this policy must retain the source/checkpoint/input identities with
each output, run only one model instance per GPU worker, and report an error or terminate on a
worker failure. It must not silently retry with a different model, precision, implementation, or
remote service. A restart is a fresh, provenance-bound checkpoint load.

## Persistent-worker measurements

The probe performed `first_pass` and `warm_pass` from the *same loaded model instance*. Raw
`depth`, `confidence`, `extrinsics`, and `intrinsics` arrays were compressed separately and then
compared exactly; elapsed time includes input processing, source inference, conversion, and output
serialization. The first call includes normal GPU warm-up, but neither pass reloads the model.

| Resolution | Checkpoint load | One view first / warm | Two views first / warm | Three views first / warm | Exact first-to-warm result |
| --- | ---: | ---: | ---: | ---: | --- |
| 252 | 11.018 s | 0.953 / 0.288 s | 0.436 / 0.426 s | 0.525 / 0.515 s | every field exact for all cases |
| 504 | 10.304 s | 1.367 / 0.699 s | 1.355 / 1.313 s | 1.429 / 1.414 s | one view exact; multi-view limitation retained |

At 252 pixels, all output fields for all cases had zero differing elements and identical hashes
between first and warm passes. Peak allocated VRAM was 9.131 GiB (one), 9.176 GiB (two), and
9.177 GiB (three), consistent with WP5.

At 504 pixels, the one-view fields were exact. The persistent worker did **not** make multi-view
output bitwise repeatable: two-view depth differed at all 282,240 elements (maximum absolute delta
`0.0110130`, mean `0.00451089`) and 6 of 24 extrinsic elements (maximum `0.000111103`); three-view
depth differed at 45,947 of 275,184 elements (maximum `0.00912476`, mean `0.00152354`). Confidence
and intrinsics remained exact; three-view extrinsics remained exact. The 504 probe therefore exits
with status 1 by design and its negative JSON result is retained.

## Retained artifacts and conclusion

| Run | Location | Result |
| --- | --- | --- |
| 252 persistent worker | `X:\3D-Spatial-Mapping-Phase2\runs\p00-wp6-native-persistent-r252-20260812` | Probe JSON plus six raw predictions; success. |
| 504 persistent worker | `X:\3D-Spatial-Mapping-Phase2\runs\p00-wp6-native-persistent-r504-20260812` | Probe JSON plus six raw predictions; expected non-exact multi-view result retained. |

This is evidence that local exact-checkpoint operation remains viable and that reusing a loaded
model safely removes repeated load overhead. It is not evidence that 504-pixel multi-view geometry
is bitwise repeatable, physically accurate, or accepted. No remote worker is warranted by this
conditional work package while the exact checkpoint is locally executable; one would still need the
same source, checkpoint, input, manifest, and output-comparison contract if later required.

WP8 must smoke-test and resolve the supporting-stack closure. The 504-pixel multi-view numerical
boundary remains subject to the later strongest-camera pilot and physical validation under D008 and
D010.

