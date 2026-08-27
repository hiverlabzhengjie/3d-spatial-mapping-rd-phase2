# XR02 Public Summary - Multi-Camera Multi-Person Tracking

## Outcome

XR02 extends the anonymous XY idea into a four-camera multi-person R&D workflow:

```text
RTSP -> MediaMTX -> supervised latest frames -> YOLO -> BoT-SORT/OSNet
     -> accepted ray/floor XY -> scene-global association -> Rerun/operator console
```

The reusable release includes typed scene/camera/local/global records, fixed-camera tracker
adapters, content-addressed embeddings, deterministic journals, same-camera cannot-link,
overlap-view deduplication, gated global assignment, explicit lifecycle/ambiguity states, bounded
reacquisition, supervised capture, MediaMTX fan-out, multi-rate/latest-only scheduling, operator
controls, Rerun publication, deterministic replay and metadata scale evaluation.

BoT-SORT is the working local profile; Deep-OC-SORT remains a challenger. Supervision is optional
behind a replaceable adapter. TrackStudio is not runtime authority.

## Evidence boundary

XR02 is accepted and closed for bounded internal R&D. The retained private office evidence ran for
about 108 seconds at 6.77 end-to-end updates per second with zero worker failures. The final v5
association policy was evaluated as a deterministic association-only derivative; it was not a new
live-v5 trial.

Independent identity/floor labels were unavailable, so XY error, IDF1, HOTA, formal ID-switch,
handoff and live-deduplication accuracy are not reported. The 40-camera result covers metadata
namespaces and scene partitioning only—no forty-stream decoder, detector, ReID, MediaMTX or GPU
capacity claim follows. This is not biometric identity, survey geometry, safety monitoring,
production readiness, high availability or an SLA.

## What is excluded

No credentials, client frames/video, model weights, embeddings, RRDs, private calibration or
facility coordinates, runtime manifests, local artifact paths or retained trial payloads are
published. The included tuning/held-out fixtures are synthetic anonymous examples.
