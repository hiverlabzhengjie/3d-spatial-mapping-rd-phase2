# P01 Public Summary - Client-input and Observability Audit

## Outcome

P01 created and validated a reusable, credential-safe observability layer for four fixed RTSP
camera identities. It provides:

- fixed camera-ID to local environment-key mapping;
- endpoint validation and redaction;
- immutable owner/stream-profile identity contracts;
- sanitized stream-profile and diagnostic-capture manifests;
- bounded, read-only RTSP preflight and packet-preserving capture;
- explicit source, acquisition and processing-time separation;
- landmark solve/held-out role separation;
- rejection behavior for malformed configuration, unsafe paths, timeouts and disconnects.

The stage was accepted as an audit, not as camera registration.

## Sanitized operational findings

All four configured sources were reachable through bounded read-only diagnostics and exposed H.264
video at 1920x1080. Short diagnostic captures and full manifests were retained in the private
artifact store and are not published.

Visual review found recurring OSD regions and substantial movable office content. Those pixels and
objects are excluded from geometric evidence. Current samples did not establish complete permanent
landmark sets or confirmed structural-overlap edges.

The audit therefore retained the connectivity graph as uncertain/unobserved and specified minimal
follow-up evidence:

- quiet-period permanent-landmark review for Cameras 1-3;
- room/view confirmation followed by landmark review for Camera 4;
- later confirmation of native/crop/dewarp status;
- facility-frame and plan-control validation in P02.

## Evidence boundary

P01 does not establish:

- camera intrinsics or distortion models;
- camera position, orientation or optical centre;
- facility world coordinates;
- accepted 2D-to-3D landmark correspondences;
- camera overlap or DA3 multi-view groupings;
- point-cloud fusion or XYZ accuracy.

Owner observations and plan annotations remain private, provenance-bound inputs. Later stages must
derive coordinates from a reviewed dimension/control network and independently validate camera
poses.

## Validation

The published P01 code passed:

```text
56 tests passed
Ruff passed
strict mypy passed for 16 source files
```

No endpoint URL, credential, client image, floor-plan derivative, raw capture or local owner
manifest is included in this public snapshot.
