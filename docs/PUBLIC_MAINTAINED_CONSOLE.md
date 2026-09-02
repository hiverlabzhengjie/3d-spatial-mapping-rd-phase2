# Maintained Multi-Scene Console

This public milestone packages the reusable application changes developed after the accepted P09
and XR02 R&D demonstrations. It is source disclosure and engineering continuity, not a release of
private scene data or a claim that the system is production-ready.

## Included capabilities

- one installable `phase2-console` entry point with validated, portable local configuration;
- scene-scoped immutable registry, artifact, camera-policy and calibration records;
- variable camera rosters for facility setup, bounded RTSP capture and static processing;
- intrinsic-group review and overlap topology as separate authorities;
- fixed-centre four-solve/two-held-out calibration workflow with explicit review states;
- scene-local joint DA3 orchestration, reversible camera membership and geometry review;
- a separate mathematical `Z=0` floor derivative and restart-safe final approval;
- Live/Recording lifecycle controls, compact anonymous telemetry and scene-update scheduling;
- explicit stale, unavailable, failed, ambiguous and unsupported states rather than silent reuse;
- read-only preflight checks for source, lock, model/runtime and configured artifact boundaries.

## Scientific and operational boundaries

No published example contains a real facility frame, camera transform, endpoint, credential,
capture, point cloud, model weight or runtime database. A managed scene derives calibration,
geometry, floor and tracking inputs only from that scene. It cannot inherit another scene's
accepted evidence.

The DA3 checkpoint remains mandatory for accepted project geometry, but its weights are not
distributed. The public code does not establish survey-grade XYZ accuracy. Live anonymous
tracking remains non-safety, non-biometric internal R&D; formal MOT accuracy, long soak and
production throughput are outside the published evidence.

Managed Live currently translates exactly four enabled scene cameras into XR02's canonical worker
slots. This is an explicit compatibility adapter, not a general variable-roster tracking claim.
Static facility, capture, calibration and reconstruction paths support a scene-defined roster.

## Dependency boundaries

The root `uv.lock` covers the maintained console and native engineering tools. DA3 and XR02 retain
separate uv projects because their validated NumPy and CUDA dependency envelopes differ. Model
source, checkpoint hashes, GPU driver, FFmpeg, MediaMTX, Rerun and accepted artifact identities
remain external authority and are validated at configured boundaries.

Start with `configs/console-profile.example.json`. Keep the local profile, `.env`, runtime stores
and generated outputs outside Git. Run `phase2-console --profile <local-profile> --preflight`
before starting the service.
