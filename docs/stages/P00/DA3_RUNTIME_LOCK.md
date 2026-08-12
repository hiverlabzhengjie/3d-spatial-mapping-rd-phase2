# P00 DA3 Core Runtime and Model Lock

**Date:** 2026-08-12  
**Status:** exact-checkpoint core and selected native supporting-stack lock established; model-policy exclusions remain active.

## Immutable model identity

| Item | Locked identity |
| --- | --- |
| Upstream source | `https://github.com/ByteDance-Seed/Depth-Anything-3` at `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| Local source cache | `X:\3D-Spatial-Mapping-Phase2\cache\upstream\Depth-Anything-3-3d835ec1` |
| Mandatory checkpoint | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` |
| Checkpoint revision | `b2359bdf726fb44ef62acca04d629dcf158053e7` |
| Local checkpoint root | `X:\3D-Spatial-Mapping-Phase2\model_weights\DA3NESTED-GIANT-LARGE-1.1-b2359bdf` |
| Weight file | `model.safetensors`, 6,759,558,100 bytes, SHA-256 `8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c` |
| Checkpoint licence | CC BY-NC 4.0; compatible with the internal non-commercial R&D scope in D009 |

The source cache was checked clean after the runs. No source patch, model conversion, smaller
checkpoint, or computation-changing compatibility patch was used.

## Comparable candidates

| Candidate | Isolated interpreter | Torch/CUDA wheel | Shared artifact path |
| --- | --- | --- | --- |
| Native Windows | `X:\3D-Spatial-Mapping-Phase2\cache\runtime_envs\native-windows-cpython311\Scripts\python.exe`, CPython 3.11.4 | `torch==2.0.1+cu118`, `torchvision==0.15.2+cu118`; Torch CUDA 11.8 | `X:\3D-Spatial-Mapping-Phase2` |
| WSL2 Ubuntu | `/home/<user>/.local/share/spatial-mapping-phase2/p00-wsl-py311/bin/python`, CPython 3.11.4 | `torch==2.0.1+cu118`, `torchvision==0.15.2+cu118`; Torch CUDA 11.8 | `/mnt/x/3D-Spatial-Mapping-Phase2` |

Both candidates executed on the same NVIDIA GeForce GTX 1080 Ti with 11 GiB reported VRAM.
Their common development lock remains `requirements/p00-native-dev.lock` (SHA-256
`a334570e7412fa3a55fdc0a4d21febefe8f322816a3a9ea47ee336d9373e08ba`).

## DA3 checkpoint-core selection

The following exact package versions are the shared direct/core selection actually used to load
the pinned DA3 source and execute the controlled public-API inference calls:

| Package | Version |
| --- | --- |
| `addict` | `2.4.0` |
| `e3nn` | `0.5.1` |
| `einops` | `0.8.2` |
| `evo` | `1.37.0` |
| `fastapi` | `0.141.1` |
| `huggingface_hub` | `1.27.0` |
| `imageio` | `2.37.4` |
| `moviepy` | `1.0.3` |
| `numpy` | `1.26.4` |
| `omegaconf` | `2.3.1` |
| `opencv-python` | `4.11.0.86` |
| `pillow` | `12.2.0` |
| `plyfile` | `1.1.3` |
| `psutil` | `7.2.2` |
| `pycolmap` | `4.1.1` |
| `requests` | `2.28.1` |
| `safetensors` | `0.8.0` |
| `scipy` | `1.17.1` |
| `torch` | `2.0.1+cu118` |
| `torchvision` | `0.15.2+cu118` |
| `trimesh` | `5.0.0` |
| `typer` | `0.27.1` |
| `uvicorn` | `0.52.1` |

`numpy==1.26.4` is required by the upstream `numpy<2` declaration. `e3nn==0.6.0` was rejected:
it expects `torch.compiler`, which is absent from the selected Torch 2.0.1 line. The source imports
`addict`, although it does not declare that package; `addict==2.4.0` is therefore an explicit part
of this lock.

## Supporting-stack closure

The selected native candidate passed the bounded GeoCalib, OpenCV, SciPy, Open3D, Rerun,
PyAV/FFmpeg and FastAPI/Uvicorn smoke in WP8. Its direct selection is tracked in
`requirements/p00-native-supporting-direct.lock`; complete dependency provenance, model-weight
hash, raw smoke artifacts and retained failures are in `WP8_SUPPORTING_STACK_RESULT.md`.

GeoCalib is pinned to source commit `97b8968e7798a66bf04fcf791fb535624241bda7` with
`kornia==0.7.4`, not the incompatible current 0.8.3 release. Rerun is pinned to 0.22.1 because
newer 0.36.0 requires NumPy 2+, incompatible with this DA3 runtime's `numpy==1.26.4` boundary.
The supporting-stack installation and exact DA3 regression did not change the selected Torch,
Torchvision, NumPy, OpenCV or SciPy versions.

## Deliberate exclusions and boundaries

- `xformers` is not installed. The unmodified DA3 source catches its absence and selects its own
  SwiGLU implementation. This is a source-provided implementation fallback, not a local patch;
  it must remain declared in every run manifest.
- `gsplat` is not installed. DA3 reports the optional-renderer warning, but no run requests
  `infer_gs=True` or a 3D Gaussian export.
- The native candidate now installs and smoke-tests source-declared `open3d`, `pillow-heif`, and
  `pre-commit`. The selected supporting stack is bounded to its tested operations; it is not
  permission to use untested UI, export, rendering or calibration-acceptance paths.
- The locks above establish the DA3 checkpoint-core route only. They are not permission to use
  untested export formats, web serving, or geometry as accepted project output.

## Dependency-closure record

After WP8, `pip check` remains non-zero only because the upstream package declares xFormers, which
is deliberately rejected under D015. The earlier native Open3D, Pillow-Heif and pre-commit gaps are
closed; the WSL2 diagnostic candidate is retained without the selected-native supporting installation:

| Candidate | Reported unsatisfied declarations |
| --- | --- |
| Native Windows | `xformers` only; explicitly prohibited under D015 |
| WSL2 Ubuntu | `pre-commit`, `xformers` |

This is retained negative evidence, not a clean environment claim. WP6 records the final allowed
optimization/runtime controls in `WP6_NATIVE_OPTIMIZATION_RESULT.md`; WP8 must close or explicitly
reject each supporting capability before P00 can offer a canonical-runtime recommendation.

## Reproduction entry point

Use the tracked harness [run_p00_da3_smoke.py](../../../scripts/run_p00_da3_smoke.py) with the
candidate-specific interpreter and path form above. It fixes the upstream fixture identities,
uses DA3's public `inference` API at a supplied `process_res`, records one/two/three-view
provenance and measurements, and can retain raw `depth`, `confidence`, `extrinsics`, and
`intrinsics` arrays as compressed NPZ files in the untracked `D:` run store.

The exact successful 504-pixel manifests and raw outputs are indexed in
`DA3_NATIVE_RUNTIME_COMPARISON.md`.

