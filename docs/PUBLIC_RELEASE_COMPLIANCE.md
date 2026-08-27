# Public Source and Third-Party Notices

This file is an engineering disclosure, not legal advice.

## Project licence and source offer

Project-owned source in this repository is released under the GNU Affero General Public License,
version 3. The repository itself is the public corresponding-source location for the published
P09/XR02 application variant. The XR02 browser console includes a link back to this source.

Private RTSP credentials, client media, facility coordinates, accepted local calibration/geometry
artifacts, model weights, recordings, embeddings and runtime outputs are data—not part of this
source snapshot—and are intentionally not distributed. The public launchers use portable local
configuration rooted at `SPATIAL_MAPPING_ARTIFACT_ROOT` rather than owner-specific paths.

## Principal third-party components

| Component | Published identity | Licence/current treatment |
| --- | --- | --- |
| Ultralytics | `8.4.123`; YOLO11n asset identity recorded separately | AGPL-3.0; imported as an external dependency; no weights redistributed |
| BoxMOT | commit `8f8babc5302024b13db7e7faeb50b3da55d1e815` | AGPL-3.0; installed metadata reported `21.0.0` despite the v22 source checkout |
| Torchreid / OSNet code | Torchreid `0.2.5` | MIT; OSNet checkpoint redistribution/use must be reviewed separately |
| Supervision | `0.30.0` | MIT; optional adapter only |
| MediaMTX | `v1.20.1`, commit `883194a` | MIT; external gateway binary, not redistributed |
| Rerun | `0.22.1` | Apache-2.0; external SDK/viewer |
| PyAV | `16.0.1` | BSD-3-Clause; external package |
| TrackStudio | `v0.1.0`, commit `69d9d8131968afe70e537990108e5e5c1afa88b8` | Apache-2.0; optional reference, not an operational dependency |
| DA3 source/model | upstream project; Nested checkpoint used privately | Upstream terms apply; no source modifications or weights redistributed here |

Upstream source and licence locations:

- <https://github.com/ultralytics/ultralytics>
- <https://github.com/mikel-brostrom/boxmot>
- <https://github.com/KaiyangZhou/deep-person-reid>
- <https://github.com/roboflow/supervision>
- <https://github.com/bluenviron/mediamtx>
- <https://github.com/rerun-io/rerun>

Anyone distributing a binary/runtime assembled from this source must preserve notices, provide the
applicable complete corresponding source and comply with every dependency/model term. A commercial
or proprietary product requires a separate licence and model-data review.
