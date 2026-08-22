# P08 Public Summary - Floor reference and integrated operator workflow

P08 added a deterministic, non-destructive facility-floor reference derived from the selected
working point cloud. The selected floor layer is a bounded mathematical plane at facility floor
level. Original point positions and colors remain unchanged, and the floor is stored separately
from camera-observed geometry.

The stage also integrated the previously separate tools into a reusable operator workflow. It
provides a configurable scene workspace and camera roster, explicit phase states, compatibility
adapters, bounded background jobs, persisted current-run state, shared service/CLI paths,
secret-safe errors and constrained launch of a verified Rerun recording.

The refined console adds one coherent camera-preparation, capture, calibration, reconstruction,
floor-processing and review path. Long operations are mutually gated; completed results are bound
to exact source identities; restarts preserve operator state; repeat preview construction is
idempotent; and scene history supports dependency-checked retention and exact-file cleanup with
protected authority records.

P08 is accepted and closed for bounded internal R&D use. The public milestone includes reusable
implementation and synthetic tests only. It intentionally excludes the private live catalog,
facility inputs, selected artifacts, deletion history and machine-local configuration.

This milestone does not establish survey-grade XYZ accuracy, as-built truth, accepted camera
connectivity, safety suitability or client-acceptance geometry. Public release excludes camera
imagery, facility coordinates, client material, credentials, machine-local paths and generated
artifacts.
