# 3D Spatial Mapping R&D - Phase 2

This repository is the public-facing engineering snapshot of an R&D project investigating
low-touch, facility-coordinate 3D reconstruction from fixed CCTV cameras.

The intended workflow registers independently placed cameras into a shared metric facility frame
using a dimensioned floor plan, permanent structural landmarks, camera mounting information,
GeoCalib and PnP-based pose estimation. Depth Anything 3 (DA3) supplies scene depth and geometry;
verified facility references remain authoritative for world position, axes and scale.

## Published milestones

### P00 - Compute feasibility

P00 established the compute and software foundation:

- verified the exact `DA3NESTED-GIANT-LARGE-1.1` checkpoint on a GTX 1080 Ti;
- compared pinned native Windows and WSL2 runtimes;
- selected native Windows for the measured local runtime;
- demonstrated provenance-bound one-, two- and three-view inference at 252 pixels;
- retained 504-pixel multi-view numerical variability as an explicit limitation;
- smoke-tested GeoCalib, OpenCV, SciPy, Open3D, Rerun, PyAV/FFmpeg and FastAPI/Uvicorn.

P00 is partially accepted. Its runtime foundation is reusable, but its synthetic fixtures are not
office geometry and do not establish real-scene XYZ accuracy.

See [the P00 handoff](docs/stages/P00/HANDOFF.md).

### P01 - Client-input and observability audit

P01 established reusable credential-safe contracts and a bounded read-only RTSP diagnostic
workflow for four fixed camera identities. The audit verified the intended profile fields and
capture provenance while keeping endpoints, credentials and media outside Git.

P01 was accepted as an audit stage. It deliberately did not accept camera calibration, poses,
landmark correspondences, overlap edges, world coordinates or geometry. The real-scene findings
remain summarized without publishing client imagery or site-specific annotations.

See [the sanitized P01 summary](docs/stages/P01/PUBLIC_SUMMARY.md).

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/spatial_mapping_phase2/` | Reusable typed contracts and policy logic |
| `tests/` | Unit tests covering success and rejection behavior |
| `scripts/` | Bounded diagnostic and compute-feasibility entry points |
| `requirements/` | Pinned P00 direct dependency selections |
| `configs/` | Non-secret configuration examples |
| `docs/` | Project plan and selected public technical evidence |

## Validation

The P01 snapshot passed:

```text
56 tests passed
Ruff passed
strict mypy passed for 16 source files
```

Model weights, raw outputs, client media, floor-plan derivatives, RTSP credentials, endpoint
values, owner-local manifests and artifact stores are not included.

## Scope and licensing

This repository describes non-commercial R&D. DA3 source and model assets are governed by their
upstream licences; the pinned Nested checkpoint is recorded as CC BY-NC 4.0. No DA3 weights are
redistributed here. Review every upstream dependency and model licence before commercial use.

Unless a separate licence file is added, the project-owned material in this repository is not
offered under an additional open-source licence.
