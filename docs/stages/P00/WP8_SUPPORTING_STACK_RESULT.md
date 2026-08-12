# P00 WP8 Native Supporting-Stack Result

**Date:** 2026-08-12  
**Selected runtime:** native Windows CPython 3.11.4 under D014  
**Result:** passed bounded stack smoke; the exact DA3 regression smoke also passed after installation.

## Selected dependency and source lock

The direct supporting-stack selections are tracked in
`requirements/p00-native-supporting-direct.lock`. The complete resolved native environment freeze
has 146 lines and SHA-256 `fd5e2bcc6259b025ff17bc7aed2c96cdd3c683a50444b779ed1f678de841cabf`:

`X:\3D-Spatial-Mapping-Phase2\runs\p00-wp8-native-stack-retry-20260812\native_runtime_freeze.txt`

| Component | Locked identity | Result |
| --- | --- | --- |
| GeoCalib source | `https://github.com/cvg/GeoCalib`, clean commit `97b8968e7798a66bf04fcf791fb535624241bda7`, Apache-2.0 code | Imported and performed a synthetic CUDA calibration. |
| GeoCalib pinhole weights | Official v1.0 release; SHA-256 `86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac` | Downloaded to `X:\3D-Spatial-Mapping-Phase2\model_weights\torch_hub\hub\geocalib\pinhole.tar`; no client media used. |
| Kornia | `0.7.4` | Compatible with selected `torch==2.0.1+cu118`. |
| OpenCV | `4.11.0.86` | Synthetic projection/PnP round-trip passed. |
| SciPy | `1.17.1` | Least-squares and rotation operations passed. |
| Open3D | `0.19.0` | Point-cloud/bounding-box operation passed. |
| Rerun SDK | `0.22.1` | Non-empty `.rrd` recording produced. |
| PyAV | `18.0.0` | Decoded a synthetic H.264 stream written by FFmpeg. |
| FFmpeg / FFprobe | `7.1-full_build-www.gyan.dev` | Encoded and inspected a 32x24, 3 fps H.264 test video. |
| Pillow-Heif | `1.5.0` | HEIF encode/decode shape and mode round-trip passed. |
| pre-commit | `4.6.2` | Module command returned its version. |
| Lightweight web stack | `fastapi==0.141.1`, `uvicorn==0.52.1` | In-process `/healthz` request and Uvicorn application loading passed. |

## Bounded smoke evidence

The successful, provenance-bound manifest and generated synthetic artifacts are retained at:

`X:\3D-Spatial-Mapping-Phase2\runs\p00-wp8-native-stack-retry-20260812`

The smoke uses no RTSP source, credential, client media, plan, or accepted geometry. It does not
make a calibration-accuracy claim. GeoCalib used a uniform synthetic RGB image of shape `3x96x128`
only to verify that the pinned source, official weights and CUDA inference path load and return the
expected finite result tensors. Its full call took 8.964 s and returned camera, gravity,
covariance, fields and uncertainty tensors.

Other bounded results were:

- OpenCV synthetic PnP reprojection maximum error: `5.684341886080802e-14` pixels.
- Open3D four-point cloud bounding-box extent: `[1.0, 1.0, 1.0]`.
- Rerun recording: 2,755 bytes, SHA-256
  `f6c82b7692953b10dcba8dca984dfa2d0e5f2145eab40b1000c49f3a3d0209a2`.
- PyAV decoded three frames from the synthetic FFmpeg H.264 stream.

FastAPI's current test client emitted a Starlette deprecation warning concerning its HTTPX use,
but the request and Uvicorn config load succeeded. This is a maintenance warning for later web
implementation, not a failed smoke result.

## Retained negative findings and resolution

| Finding | Handling |
| --- | --- |
| Unconstrained `rerun-sdk==0.36.0` requires NumPy 2+, conflicting with DA3's tested `numpy==1.26.4`. | Rejected. A constrained resolver search found `rerun-sdk==0.22.1` compatible with NumPy 1.26.4; it passed the recording smoke. NumPy/Torch were not upgraded. |
| GeoCalib's unconstrained `kornia==0.8.3` failed at import under Torch 2.0.1 because its JIT code passes a tuple dimension to `Tensor.any`. | Retained in the first run manifest. `kornia==0.7.4` imported and enabled the unchanged GeoCalib source/weights smoke; Torch remained `2.0.1+cu118`. |
| Normal Open3D installation failed while unpacking a long Jupyter-widget asset because Windows long paths are disabled. | Retained. A temporary `S:` mapping to the *existing* isolated runtime shortened only the installation path and was removed immediately. No Windows policy, source, artifact layout, or model computation changed. |
| The first smoke harness read a non-existent `pre_commit.__version__`. | Retained in the first manifest, corrected to package metadata in commit `6407832`, then rerun successfully. |
| `pip check` is nonzero only for `xformers`. | Expected and retained under D015: xFormers is an unvalidated accelerator and remains prohibited. Open3D, Pillow-Heif and pre-commit are now installed. |

The initial failed run is retained at
`X:\3D-Spatial-Mapping-Phase2\runs\p00-wp8-native-stack-20260812`.

## Deferred camera-distortion evaluation

GeoCalib's pinned source exposes `pinhole`, `simple_radial`, `radial`, and
`simple_divisional` camera models. WP8 intentionally downloaded and smoke-tested only the official
`pinhole` weights; it did **not** download, pin, or validate the separate official `distorted`
weights. This is not evidence that a wide-angle CCTV camera should use any particular distortion
model.

P04 must pin and test the applicable official weights, compare plausible pinhole/radial candidates
on clean camera frames, and select a model using shared-intrinsic stability and held-out residuals
under the native stream profile. The result must remain provenance-bound and must not be inferred
from the camera being wide angle alone.

## Exact DA3 regression after stack installation

After the successful stack smoke, the mandatory checkpoint was reloaded from its existing local
path and completed the established 252-pixel one-, two-, and three-view cases. All `depth`,
`confidence`, `extrinsics`, and `intrinsics` arrays were finite. Peak allocated VRAM was 9.107,
9.142 and 9.147 GiB respectively; load time was 8.710 s CPU plus 1.759 s transfer. Raw outputs and
the manifest are retained at:

`X:\3D-Spatial-Mapping-Phase2\runs\p00-wp8-da3-regression-20260812`

This regression validates that the selected supporting stack did not break local exact-checkpoint
execution. It does not supersede the WP5/WP6 repeatability findings, enable xFormers/`gsplat`, or
authorize accepted geometry.

## Conclusion

The native candidate now has a tested supporting stack for the intended Phase 2 calibration,
geometry, evidence, media, and lightweight-web paths. The source/model policy remains unchanged:
DA3 source-selected FP16/SwiGLU fallback is retained, xFormers and `gsplat` remain uninstalled,
and the multi-view 504-pixel numerical boundary remains open for later physical validation.

