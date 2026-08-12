# Phase 2 Detailed R&D Plan

## Operating model

Build a low-touch, repeatable system that captures four fixed office CCTV streams over RTSP,
registers each camera into a metric floor-plan coordinate system, reconstructs static geometry
with DA3 Nested Giant-Large 1.1, and presents accepted evidence through a local workflow console
and Rerun.

The floor plan, verified dimensions, permanent landmarks, camera heights and accepted poses are
the spatial authority. DA3 is the geometry engine. World registration does not require common
view and does not authorize cloud fusion without validation.

Every stage uses a reviewed implementation brief, bounded work packages, versioned inputs,
automated tests, visual evidence where relevant, an append-only decision record and a human gate.

## P00 - Project foundation and compute feasibility

**Objective:** Prove the mandatory model and supporting stack can run reproducibly before the
project depends on them.

**Inputs:** Windows workstation, GTX 1080 Ti, Ubuntu WSL2, exact DA3 checkpoint, Phase 1 records
and the dedicated 4 TB drive.

**Work packages:**

1. Verify and mount the large-artifact store without formatting or overwriting it implicitly.
2. Establish isolated code, secret, model, capture, cache, run and export locations.
3. Compare native Windows and WSL2 with the same pinned smoke suite.
4. Test exact DA3 model loading and one-, two- and three-view inference at controlled resolutions.
5. Record output validity, repeatability, VRAM, RAM and runtime.
6. Permit safe inference optimizations; reject workarounds that alter learned computation.
7. If local execution fails, validate a remote GPU worker with the same checkpoint and code.
8. Smoke-test GeoCalib, OpenCV, SciPy, Open3D, Rerun, PyAV/FFmpeg and the web stack.
9. Audit Phase 1 code and port only applicable, tested contracts or modules.

**Outputs:** Canonical runtime, dependency/model lock, compute benchmark, storage/secrets policy,
remote-worker contract if required, and Phase 1 reuse assessment.

**Gate:** Exact checkpoint completes required tests locally or remotely with finite, repeatable,
provenance-bound outputs and no correctness-altering compatibility patch.

## P01 - Client-input and observability audit

**Objective:** Determine whether each camera can be calibrated and registered with the current
low-touch inputs.

**Inputs:** Four RTSP sources, camera heights and approximate plan positions, camera/stream
information, quiet-period frames and the scanned plan.

**Work packages:**

1. Assign stable camera IDs and bind them to physical devices and stream profiles.
2. Record resolution, codec, FPS, rotation, crop, overlays, dewarping and stability.
3. Capture short concurrent diagnostic clips without changing the live-source design.
4. Identify OSD and dynamic exclusion regions.
5. Inventory permanent landmarks, targeting 6-10 solving points and at least two held-out points
   per camera with broad image and depth coverage.
6. Produce a masked structural-overlap graph and a site-effort ledger.

**Outputs:** Client-input manifest, camera/stream profiles, diagnostic captures, observability
matrix, preliminary connectivity graph and minimal missing-input list.

**Gate:** Every camera has a verified stream profile, mounting information and a viable landmark
set or a specific minimal action required to obtain one.

## P02 - Facility frame and scanned-plan digitization

**Objective:** Convert the scanned plan into a reproducible metric world reference without
treating scan pixels as ground-truth coordinates.

**Inputs:** Original scan, printed dimension chains, ceiling height, permanent features and
minimum independent physical checks.

**Work packages:**

1. Preserve and hash the source scan.
2. Define a reviewed right-handed metre-based frame with floor at Z=0.
3. Digitize dimension-chain intersections and permanent structural features.
4. Derive coordinates from verified dimensions and fit a plan-pixel/world display transform.
5. Retain local scan-distortion and transform residuals.
6. Build the landmark database with feature meaning, source and uncertainty.
7. Verify all camera heights, ceiling height and one horizontal distance in both the overlapping
   and isolated camera areas; add measurements only when later evidence requires them.

**Outputs:** World reference, georeferenced plan, control network, landmark database, plan
uncertainty record and spot-check ledger.

**Gate:** World axes and units are unambiguous, and dimension chains plus spot checks form a
consistent control network within recorded uncertainty.

## P03 - Live capture and workflow-console foundation

**Objective:** Establish reliable live ingestion and reproducible capture selection before model
integration.

**Inputs:** Accepted stream profiles, local credentials and artifact-store configuration.

**Work packages:**

1. Implement bounded RTSP connect, timeout, reconnect and shutdown behavior.
2. Preserve stream PTS, local monotonic acquisition time and stream-profile versions.
3. Support concurrent on-demand four-camera capture, short MP4 caches and selected frames.
4. Create capture bundles using closest compatible timestamps and report actual skew.
5. Build localhost stream-health, preview, capture, session-browser and job-status pages.
6. Expose equivalent CLI operations through the same service layer.
7. Add a local RTSP interruption/reconnect fixture.

**Outputs:** RTSP adapter, immutable capture-session format, cached-media workflow, initial web
console, CLI and RTSP integration tests.

**Gate:** Four feeds can be monitored, captured and reconnected without silent loss, and a
selected static bundle is reproducible from its manifest.

## P04 - Strongest-camera calibration and registration pilot

**Objective:** Prove the low-touch method on the best-observed camera and derive provisional
acceptance thresholds.

**Inputs:** World reference, pilot frames, camera height/location, lens/FOV prior and structural
landmarks.

**Work packages:**

1. Run multi-frame GeoCalib candidates across plausible pinhole/radial models.
2. Evaluate shared-intrinsic stability and bind results to the native stream profile.
3. Annotate solve/held-out landmarks in the console and repeat a subset for click uncertainty.
4. Solve PnP/RANSAC candidates and refine with robust reprojection, gravity, height, mount and
   physical-orientation constraints.
5. Validate held-out projections, leave-one-out sensitivity, multi-frame repeatability, frustum
   placement and physical viewing direction.
6. Retain DA3 unposed estimates as diagnostics only.
7. If weak, add clean frames and bounded refinement, then allow a brief in-situ board capture.
8. Derive and freeze provisional office-pilot tolerances at human review.

**Outputs:** Pilot intrinsics, accepted pose, convention tests, held-out diagnostics, escalation
record and frozen provisional acceptance policy.

**Gate:** The pose is supported independently of its solving landmarks and agrees with the plan,
height, gravity and physical view.

## P05 - Four-camera registration and connectivity

**Objective:** Register all cameras independently and establish evidence-based DA3 groupings.

**Inputs:** Frozen pilot procedure, all camera observations, world reference and overlap evidence.

**Work packages:**

1. Apply the calibration/pose workflow independently to Cameras 2-4.
2. Preserve competing hypotheses until held-out validation selects one.
3. Use staged calibration-board fallback only for cameras that need it.
4. Compare overlapping cameras through shared permanent structures.
5. Classify poses as accepted, provisional or rejected with explicit reasons.
6. Finalize the connectivity graph, independently testing Camera 3 edges and treating Camera 4
   as isolated unless evidence changes that conclusion.
7. Present poses, frustums and diagnostics in the console and Rerun.

**Outputs:** Versioned camera registry, validation packages, plan/frustum view, final overlap graph
and site-effort summary.

**Gate:** Full success targets four accepted poses. Weak cameras remain provisional/unregistered;
partial stage closure requires explicit review.

## P06 - DA3 reconstruction-mode evaluation

**Objective:** Select the best defensible DA3 mode for each validated component using the exact
mandatory checkpoint.

**Inputs:** Camera registry, static bundles, connectivity graph, model runtime and exclusion masks.

**Work packages:**

1. Produce single-view DA3 baselines for every camera.
2. Run pose-conditioned multi-view DA3 for Cameras 1-2.
3. Compare Camera 3 in the three-camera group, its strongest pair and single-view mode.
4. Select Camera 3's mode by confidence, depth consistency, structural residuals and impact on
   the strong component.
5. Process Camera 4 in single-view metric mode with its validated pose.
6. Preserve raw depth, confidence, processed images, predicted intrinsics/extrinsics and exact
   runtime/model provenance.
7. Do not treat repeated fixed-camera timestamps as additional geometric viewpoints.

**Outputs:** Raw DA3 artifacts, mode comparison, selected policy per component and performance
report.

**Gate:** Selected modes are finite, repeatable and metric; Camera 3 must help or avoid material
degradation, otherwise it remains a separate patch.

## P07 - World-space geometry and controlled fusion

**Objective:** Create usable facility-frame geometry without inventing missing connections.

**Inputs:** Raw depth/confidence, accepted intrinsics/poses, world reference, masks and selected
component policies.

**Work packages:**

1. Back-project with exact processed intrinsics and transform through accepted world poses.
2. Preserve raw camera-space and world-space clouds before correction or fusion.
3. Apply versioned confidence, bounds, dynamic-mask, downsampling and outlier policies.
4. Evaluate floor, ceiling, wall, landmark and overlap consistency.
5. Investigate systematic DA3 scale/alignment bias against independent references.
6. Allow a derived correction only with sufficient evidence; preserve raw geometry and residuals.
7. Use established Open3D operations for supported filtering, comparison and fusion.
8. Fuse only validated overlap and retain isolated/weak patches separately.
9. Leave unobserved regions explicit and record every transform/filter in manifests.

**Outputs:** Raw and filtered per-camera patches, validated component clouds, controlled combined
scene, explicit gaps/rejections and raw-versus-derived comparison.

**Gate:** Transform checks pass, patches occupy plausible plan locations and every fused region
passes structural/overlap validation.

## P08 - XYZ accuracy and robustness validation

**Objective:** Quantify what can be defended about camera pose, geometry and world XYZ.

**Inputs:** Accepted poses/geometry, held-out landmarks, spot checks, repeated captures and
validated overlap.

**Work packages:**

1. Report pose evidence separately from depth/geometry evidence.
2. Evaluate held-out reprojection, height, gravity, mounting and structural residuals.
3. Compare reconstructed distances with independent spot checks.
4. Measure overlap disagreement and repeatability across quiet capture sessions.
5. Run landmark-removal, annotation-perturbation and camera-removal sensitivity tests.
6. Exercise wrong landmarks, weak distribution, stream-profile change, camera movement, weak
   overlap, OOM and local/remote worker interruption.
7. Report distributions and regional evidence, not one unsupported global accuracy value.

**Outputs:** Validation dataset, error metrics, sensitivity results, accuracy envelope, accepted
office configuration and actual site-effort evidence.

**Gate:** Every accuracy statement is backed by held-out/independent references, and failures are
detectable as provisional or rejected states.

## P09 - Integrated demonstration and handoff

**Objective:** Deliver the repeatable workflow, interactive evidence and audience-specific
documentation.

**Inputs:** Accepted services, policies, camera registry, geometry and validation results.

**Work packages:**

1. Complete the workflow console for setup, health, capture, plan/landmarks, calibration review,
   DA3 jobs, validation and artifacts.
2. Ensure each UI operation has an equivalent command/configuration and run manifest.
3. Build the final Rerun recording with plan, cameras, landmarks, patches, raw/derived geometry,
   diagnostics, gaps and rejections.
4. Execute and verify a fresh live run and a cached-data reproduction run.
5. Produce product-manager and engineering reports in Markdown and PDF.
6. Deliver installation, operator, calibration, recovery and reproduction guides.
7. Close stages with reviewed handoffs, commits and private-remote verification.
8. Prepare a sanitization checklist without publishing automatically.

**Outputs:** Pipeline, workflow console, CLI, plan/camera view, geometry patches, Rerun artifact,
two report packages, guides and final private project history.

**Gate:** A new operator can reproduce the cached workflow, a live capture produces traceable
outputs, and every presented coordinate or fused region has inspectable provenance.

## Cross-stage test policy

- Unit-test schemas, projections, transforms, PnP, refinement, time identity and graph policy.
- Use synthetic geometry to verify convention and rejection behavior.
- Integration-test RTSP interruption, local/remote worker compatibility and UI/CLI parity.
- Test strong, partial and absent overlap; quiet and cluttered captures; wrong landmarks; stream
  changes; moved cameras; compute failures; and unsupported fusion.
- Visually inspect plan registration, camera frustums, geometry and final documents.
- Preserve failed experiments and negative findings with sufficient provenance.


