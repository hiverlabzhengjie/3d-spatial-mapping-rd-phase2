# P00 WP9 Phase 1 Reuse Assessment

**Date:** 2026-08-12
**Result:** one small capture-time contract ported and tested; all other inspected material is
deferred or rejected from P00 to avoid importing a previous project's runtime or later-stage
assumptions.

## Audited Phase 1 snapshot

P00 cloned the documented Phase 1 public source into the isolated upstream cache and inspected it
read-only. The snapshot was clean at commit
`e9e038c498c17d83bcfa65a71e1c355e3f4aa8d7` (2026-08-10T14:39:04+08:00), from
`https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`. It remains outside the Phase 2
Git worktree at:

`X:\3D-Spatial-Mapping-Phase2\cache\upstream\phase1-3d-spatial-reconstruction-rd`

No Phase 1 media, weights, recordings, `.env` values, generated output, vendor checkout, or model
lock was copied into Phase 2. The audit treats Phase 1 as project-owned background reference under
D001, not as a runtime baseline.

## Result by candidate

| Phase 1 material | WP9 outcome | Reason and Phase 2 boundary |
| --- | --- | --- |
| `ingestion/sources.py::TimestampTransform` | **Ported** as `spatial_mapping_phase2.capture_time.TimestampTransform` | Dependency-free, small and clear affine PTS-to-capture-time mapping. Its source timestamp, capture timestamp, processing time and model-completion time remain explicitly separate. New Phase 2 success/rejection tests passed. |
| `contracts.py::FrameIdentity` and `SynchronizedFrameBundle`; `ingestion/synchronization.py` | Deferred to P03 | Their immutable identity and capture-order principles are reusable, but their fields bind prior-project session, pose and synchronization semantics. P03 must instead define the Phase 2 RTSP/session/stream-profile contract, including source PTS, local monotonic acquisition time, actual skew and credential-free references. |
| `ingestion/sources.py` RTSP/PyAV source, reconnect code and local MediaMTX configuration | Deferred to P03 | P00 must not introduce RTSP endpoints, credentials or stream-profile assumptions. P03 owns bounded connect/reconnect behavior and its local interruption fixture. |
| `geometry/transforms.py` and pose/calibration helpers | Deferred to P02/P04 | The routines are understandable and their explicit `T_target_from_source` naming is a useful reference, but the facility world axes, units, control network, camera convention and held-out validation have not yet been established. P02/P04 must port or reimplement only after those contracts exist and add their own synthetic and physical checks. |
| `config.py`, DA3 MPS adapter and MPS diagnostics | Rejected for P00 reuse | They encode the prior macOS/MPS, YOLO and Qwen baseline. P00 uses the native-Windows exact-checkpoint lock and must not import a different model/runtime policy. |
| Static reconstruction, perception, localization, interaction, orchestration and Rerun-presentation modules | Deferred to their owning P03-P07 stages | They depend on Phase 1 scene-specific assets and/or contracts not yet proved for the office pilot. No code is ported before the relevant stage can test it against Phase 2 evidence. |

## Ported contract and test coverage

`src/spatial_mapping_phase2/capture_time.py` adapts only the affine timestamp mapping concept. It
does not decode media, authenticate to RTSP, synchronise cameras, fabricate a missing timestamp, or
claim a clock transform has been calibrated. `tests/test_capture_time.py` covers identity and
non-identity mappings plus zero/negative/non-finite scale, non-finite offset, non-finite source
time, and negative resulting capture time.

The selected native CPython 3.11.4 environment completed the full current check set after the port:

```text
43 passed
ruff check src tests: passed
mypy src tests: passed (14 source files)
```

Two pre-existing P00 test-only type errors were corrected while running the required full type
check: dynamic untyped `dataclasses.replace` use in the native-policy rejection matrix and indexing
an `object` returned by a JSON-ready manifest dictionary. The test meaning and runtime behaviour
were unchanged.

## Reuse constraints for later stages

Phase 1 principles worth retaining are explicit time authority, immutable provenance, raw-versus-
derived separation, and preserving unavailable evidence as unavailable. They do not authorize
copying Phase 1's scene-dependent thresholds, calibration numbers, camera poses, raw artifacts,
model variants, or Mac-specific execution path. Every later reuse must identify the Phase 1 source
commit, state its Phase 2 applicability, and add Phase 2 tests before becoming a dependency.

