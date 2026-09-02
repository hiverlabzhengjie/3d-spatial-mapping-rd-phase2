# Third-party and model notices

This is an engineering disclosure for the currently selected internal R&D stack, not legal advice.
The repository's project-owned source is licensed under AGPL-3.0 as recorded by D084. Dependencies,
external binaries and model weights remain governed by their own terms.

## Maintained combined-console runtime

| Component | Selected identity | Terms / boundary |
| --- | --- | --- |
| FastAPI | 0.141.1 | MIT |
| Uvicorn | 0.52.1 | BSD-3-Clause |
| NumPy | 1.26.4 in the native console runtime | BSD-3-Clause; wheel contains separately disclosed numerical runtimes |
| SciPy | 1.17.1 | BSD-3-Clause; wheel contains separately disclosed numerical runtimes |
| OpenCV Python | 4.11.0.86 | Apache-2.0 |
| PyMuPDF | 1.24.1; optional `facility` extra | AGPL-3.0 or separately licensed commercial terms |
| PyAV | 18.0.0 in the native runtime; optional `live-capture` extra | BSD-3-Clause; FFmpeg build terms must also be reviewed for redistribution |
| Rerun SDK/viewer | 0.22.1 | Installed metadata: MIT OR Apache-2.0 |
| GeoCalib | source commit `97b8968e7798a66bf04fcf791fb535624241bda7` | Apache-2.0 |

## DA3 geometry runtime

| Component/model | Selected identity | Terms / boundary |
| --- | --- | --- |
| Depth Anything 3 source | ByteDance-Seed commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | Apache-2.0 |
| DA3 Nested Giant-Large 1.1 checkpoint | revision `b2359bdf726fb44ef62acca04d629dcf158053e7`; SHA-256 recorded in P00/P06 | CC BY-NC 4.0 as recorded at acquisition; weights are not redistributed. The non-commercial restriction blocks unreviewed commercial use even though project source is AGPL-3.0. |
| PyTorch / Torchvision | native DA3 runtime 2.0.1+cu118 / 0.15.2+cu118 | BSD-style upstream terms; CUDA/NVIDIA components retain their own terms |

## P09/XR02 tracking runtime

| Component/model | Selected identity | Terms / boundary |
| --- | --- | --- |
| Ultralytics YOLO | 8.4.123; external YOLO11n weight hash recorded in P09/XR02 | AGPL-3.0 path selected under D049/D084; no Enterprise licence or weight redistribution is claimed |
| BoxMOT | source v22.0.0 commit `8f8babc5302024b13db7e7faeb50b3da55d1e815`; installed metadata 21.0.0 | AGPL-3.0 |
| OSNet x0.25 MSMT17 checkpoint | SHA-256 `6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99` | Torchreid code is MIT; checkpoint redistribution and dataset-derived terms remain unresolved. Weight is not redistributed. |
| Supervision | 0.30.0 optional overlay | MIT |
| MediaMTX | v1.20.1 commit `883194a` | MIT; exact Windows binary provenance is recorded in XR02 |
| GStreamer diagnostic overlay | 1.28.6 diagnostic-only bundle | Mixed LGPL/GPL/restricted plugin distribution; not an approved shipping bundle |

The exact transitive locks, hashes and source/model provenance remain in the P00, P09 and XR02
evidence packages. Before distributing a binary, hosted service or commercial variant, reproduce
the dependency SBOM, include required licence texts/notices and source offer, resolve the DA3/OSNet
model terms, and either comply with AGPL network/source obligations or replace/acquire suitable
components. Do not infer that one component's licence grants rights to client media, building data,
model training data or a model checkpoint.
