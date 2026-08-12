# Phase 2 Project Charter

## Purpose

Build a repeatable, low-touch workflow that places four fixed office CCTV cameras and their
DA3-derived static geometry into one metric facility coordinate system.

## Pilot topology

- Cameras 1 and 2 have strong overlap.
- Camera 3 has useful but weaker overlap with Cameras 1 and 2.
- Camera 4 is effectively independent.
- Live RTSP is the operational source; cached MP4 clips and extracted frames provide
  reproducible R&D snapshots.

## Priority order

1. Usable world-aligned geometry and correct camera placement on the plan.
2. A demonstrable end-to-end workflow.
3. Validated XYZ accuracy.
4. Minimum site effort, increased only when evidence demonstrates need.

## Governing spatial principle

The scanned floor plan, verified dimensions, permanent structural landmarks, camera mounting
information and accepted camera poses define the facility world frame. DA3 supplies depth and
geometry. DA3 estimates may be retained as diagnostics but do not silently redefine accepted
world position, axes, direction or scale.

World registration does not imply point-cloud fusion. Non-overlapping cameras may produce
separate geometry patches expressed in the same facility frame. Gaps remain visible unless
measured evidence supports them.

## Intended technical approach

- Digitize and verify the scanned floor plan into a right-handed metre-based world frame.
- Ingest and health-check all four RTSP streams; create immutable capture sessions.
- Estimate intrinsics and gravity from several clean frames, initially using GeoCalib.
- Solve camera world poses from permanent 2D-to-3D landmarks with OpenCV PnP/RANSAC.
- Refine and validate poses using established optimization, height, gravity and plan evidence.
- Run DA3 Nested Giant-Large 1.1 in pose-conditioned multi-view mode for validated connected
  components and single-view metric mode for validated isolated cameras.
- Back-project, transform, filter and validate geometry with NumPy/OpenCV/Open3D.
- Fuse only validated overlap; display all accepted patches together in Rerun.
- Operate through a small localhost workflow console backed by reusable Python services and a
  reproducible CLI. Rerun remains the 3D evidence viewer.

## Deliverables

- Repeatable technical pipeline and sanitized public engineering snapshot.
- Local workflow console and CLI.
- Registered camera/floor-plan view.
- Validated static point-cloud patches and controlled combined scene.
- Rerun inspection artifact.
- Product-manager report in Markdown and PDF.
- Engineering report in Markdown and PDF.
- Installation, operator, calibration, recovery and reproduction documentation.

## Evidence boundaries

- The project is exploratory and does not begin with a survey-grade accuracy claim.
- Acceptance tolerances will be derived from the strongest-camera pilot, annotation
  repeatability, available plan accuracy and minimal independent spot checks.
- A weak camera or geometry patch remains provisional or rejected.
- Raw and derived outputs remain separately inspectable.
- The optional mobile bridge/HLoc route is deferred, but interfaces must allow future pose
  sources without redesigning provenance.
- Dynamic detection, tracking and semantic reasoning are future work. Source-frame identity,
  timestamps, bounded workers and spatial-authority separation must support those additions.

## Fixed assumptions

- Office test environment with four fixed-lens cameras.
- Current plan source is a scanned, dimensioned PDF rather than CAD/vector data.
- On-demand static reconstruction from live capture, not continuous DA3 processing.
- Internal non-commercial R&D; commercial reuse requires a separate licence decision.
- DA3 Nested Giant-Large 1.1 is mandatory for accepted geometry.
- Native Windows and WSL2 are both evaluated; a remote GPU worker is permitted if the local
  GTX 1080 Ti cannot run the exact model correctly.
- Large artifacts will use the healthy 4 TB drive after its mount/path is verified.
- The new repository is private; any public release requires sanitization review.


