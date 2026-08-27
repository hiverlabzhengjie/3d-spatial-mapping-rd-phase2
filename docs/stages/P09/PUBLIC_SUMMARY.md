# P09 Public Summary - Live Anonymous Person XY Demonstrator

## Outcome

P09 added a bounded live demonstrator that detects a person in fixed-camera frames, estimates a
bottom-centre/visible-torso foot-point proxy, intersects the calibrated camera ray with the accepted
facility floor and reports anonymous facility-frame XY evidence. It uses latest-frame processing,
explicit drops and honest `unknown`, `ambiguous` and `multi_person_unsupported` states; dynamic DA3
inference is not part of the live loop.

The project-owned release includes the detector boundary, CUDA-only YOLO adapter, projection,
single-person fusion, live worker/service, Rerun presentation, verification logic and synthetic
tests. RTSP credentials, office media, accepted private calibration/geometry inputs, model weights,
RRDs and trial manifests are excluded.

## Acceptance boundary

P09 is accepted only as an internal-R&D, anonymous single-person demonstrator. The retained office
trial completed 252 ticks with zero worker failures, but its private artifacts are not published.
The result inherits an owner-observed approximate one-metre predecessor usability envelope; it does
not establish survey accuracy, safety use, persistent identity, multi-person tracking, production
monitoring or a service-level objective.

## Open-source boundary

The published implementation uses the Ultralytics AGPL-3.0 route and this repository is therefore
released under AGPL-3.0. Model weights are not redistributed. Users must obtain and review the
upstream model assets separately and supply their exact local paths/hashes.
