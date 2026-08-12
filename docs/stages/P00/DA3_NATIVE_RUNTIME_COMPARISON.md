# P00 DA3 Native Windows and WSL2 Comparison

**Date:** 2026-08-12  
**Scope:** WP4 controlled exact-checkpoint loading and one-, two-, and three-view inference.

## Method

The common harness is [run_p00_da3_smoke.py](../../../scripts/run_p00_da3_smoke.py). Both
candidates used the immutable source, checkpoint, Torch wheel, checkpoint hash, and three upstream
fixture sequences recorded in `DA3_RUNTIME_LOCK.md`. Calls use DA3's public `inference` API with
`process_res=504`, its documented default, `process_res_method="upper_bound_resize"`, no export
directory, and no Gaussian-splatting request. The public API exposes no `batch_size` argument;
each view-count case is one joint inference call.

The one/two-view inputs are the upstream `SOH` pair. The three-view case appends upstream
`da3_radar.png`. It deliberately tests count handling, not scene geometry: the image shapes differ
and DA3 centre-crops the three-view processed batch to `(182, 504)`. It must not be presented as a
same-scene geometry result.

| Runtime | Result manifest | Raw-output directory |
| --- | --- | --- |
| Native Windows | `X:\3D-Spatial-Mapping-Phase2\runs\p00-wp4-native-common-harness-20260812\native_standard_resolution_result.json` | same directory; `native_standard_resolution_result_*_raw_prediction.npz` |
| WSL2 Ubuntu | `X:\3D-Spatial-Mapping-Phase2\runs\p00-wp4-wsl-standard-resolution-20260812\wsl_standard_resolution_result.json` | same directory; `wsl_standard_resolution_result_*_raw_prediction.npz` |

## Exact checkpoint load

| Runtime | CPU load | GPU transfer | VRAM allocated after load | RSS after load |
| --- | ---: | ---: | ---: | ---: |
| Native Windows | 8.715 s | 1.839 s | 6.358 GiB | 0.604 GiB |
| WSL2 Ubuntu | 45.101 s | 1.760 s | 6.358 GiB | 0.748 GiB |

Both loads used `DepthAnything3.from_pretrained()` only with the local mandatory checkpoint path
and then transferred it to CUDA successfully.

## Controlled default-resolution results

Every reported depth, confidence, extrinsic, and intrinsic value was finite. The output shapes
matched the requested view count and DA3's documented crop behavior.

| Case | Output depth shape | Native elapsed / peak allocated VRAM / RSS | WSL2 elapsed / peak allocated VRAM / RSS |
| --- | --- | --- | --- |
| One view | `(1, 280, 504)` | 1.126 s / 9.223 GiB / 0.893 GiB | 2.520 s / 9.221 GiB / 1.075 GiB |
| Two views | `(2, 280, 504)` | 1.202 s / 9.371 GiB / 0.921 GiB | 3.233 s / 9.372 GiB / 1.120 GiB |
| Three views | `(3, 182, 504)` | 1.343 s / 9.370 GiB / 0.966 GiB | 2.798 s / 9.370 GiB / 1.181 GiB |

At this controlled resolution, native Windows is roughly 5.2x faster to load and 2.1-2.7x faster
per inference case while using essentially the same PyTorch-tracked VRAM. This is a provisional
performance preference only, not a P00 acceptance recommendation.

## Retained failures and repeatability findings

- A first native preflight used the former fixture location `assets/images/SOH`; it failed before
  model loading because the pinned source stores the pair under `assets/examples/SOH`.
- A first native inference call supplied `batch_size=1`. The pinned public API rejects that
  keyword; it was removed rather than patched around. All successful calls use the inspected
  source signature.
- The source's eager import of export utilities required `moviepy` and its model import required
  undeclared `addict`. These were recorded in the core lock; no vendor source was changed.
- The initial native default-resolution repeat had matching shapes and finite values but no
  matching output hashes. This failure is retained in
  `p00-wp4-native-standard-resolution-20260812/native_standard_resolution_repeat_result.json`.
- Enabling PyTorch deterministic algorithms without setting `CUBLAS_WORKSPACE_CONFIG` failed with
  the explicit CUDA/cuBLAS reproducibility error. That retained failure is in
  `p00-wp4-native-deterministic-repeat-20260812/native_deterministic_repeat_result.json`.
- With `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms enabled, cuDNN deterministic,
  and cuDNN benchmarking disabled, one-view output was bitwise identical across two fresh loads.
  Two- and three-view confidence/intrinsics remained bitwise identical, but depth (and two-view
  extrinsics) did not. The raw NPZ comparisons show two-view depth maximum/mean absolute deltas of
  `0.0423164` / `0.0144266` and three-view depth deltas of `0.0005760` / `0.0000962` in DA3's
  output scale. No physical unit or geometry tolerance is asserted at P00.

The multi-view repeatability result is a WP5 evidence item and remains open. It prohibits claiming
bitwise-reproducible multi-view accepted geometry on this runtime. It does not invalidate the
successful exact-checkpoint loading or finite controlled execution proof.

## WP4 conclusion

The mandatory checkpoint runs locally, unmodified, on both candidates at the source-default
resolution with controlled one-, two-, and three-view inputs. A remote worker is not currently
needed to establish local execution feasibility. P00 must still complete WP5-WP9, including a
repeatability acceptance/tolerance policy, failure-bound testing, supporting-stack smoke tests,
and the selective Phase 1 reuse assessment before asking the control tower to review the stage.

