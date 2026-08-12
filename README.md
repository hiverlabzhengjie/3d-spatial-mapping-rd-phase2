# 3D Spatial Mapping R&D â€” Phase 2

This repository is the public-facing engineering snapshot of an R&D project investigating
low-touch, facility-coordinate 3D reconstruction from fixed CCTV cameras.

The intended workflow registers independently placed cameras into a shared metric facility frame
using a dimensioned floor plan, permanent structural landmarks, camera mounting information,
GeoCalib and PnP-based pose estimation. Depth Anything 3 (DA3) supplies scene depth and geometry;
verified facility references remain authoritative for world position, axes and scale.

## Current public milestone: P00

P00 established the compute and software foundation:

- verified the exact `DA3NESTED-GIANT-LARGE-1.1` checkpoint on a GTX 1080 Ti;
- compared pinned native Windows and WSL2 runtimes;
- selected native Windows for the measured local runtime;
- demonstrated provenance-bound one-, two- and three-view inference at 252 pixels;
- retained 504-pixel multi-view numerical variability as an explicit limitation;
- smoke-tested GeoCalib, OpenCV, SciPy, Open3D, Rerun, PyAV/FFmpeg and FastAPI/Uvicorn;
- added tested runtime, artifact-layout, timing and measurement contracts.

P00 is partially accepted. Its runtime foundation is reusable, but its synthetic fixtures are not
office geometry and do not establish real-scene XYZ accuracy. In particular, 504-pixel multi-view
output remains experimental until later held-out and physical validation establishes a reviewed
criterion.

See [the P00 handoff](docs/stages/P00/HANDOFF.md) for the precise evidence boundary.

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/spatial_mapping_phase2/` | Reusable typed Python contracts and policy logic |
| `tests/` | Unit tests covering success and rejection behavior |
| `scripts/` | P00 smoke, measurement and supporting-stack entry points |
| `requirements/` | Pinned P00 direct dependency selections |
| `configs/` | Non-secret configuration examples |
| `docs/` | Project plan and selected public P00 technical evidence |

## Validation

The accepted P00 snapshot passed:

```text
43 tests passed
Ruff passed
mypy passed for 14 source files
```

Model weights, raw outputs, client media, RTSP credentials and local artifact stores are not
included. Paths in the documentation are portable placeholders rather than the original
workstation paths.

## Scope and licensing

This repository describes non-commercial R&D. DA3 source and model assets are governed by their
upstream licences; the pinned Nested checkpoint is recorded as CC BY-NC 4.0. No DA3 weights are
redistributed here. Review every upstream dependency and model licence before commercial use.

Unless a separate licence file is added, the project-owned material in this repository is not
offered under an additional open-source licence.

