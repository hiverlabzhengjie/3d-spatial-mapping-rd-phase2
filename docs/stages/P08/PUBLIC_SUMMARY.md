# P08 Public Summary - Floor reference and integrated workflow foundation

P08 added a deterministic, non-destructive facility-floor reference derived from the selected
working point cloud. The selected floor layer is a bounded mathematical plane at facility floor
level. Original point positions and colors remain unchanged, and the floor is stored separately
from camera-observed geometry.

The stage also established a reusable integration foundation for the previously separate workflow
tools. It provides a configurable scene workspace and camera roster, explicit phase states,
compatibility adapters, bounded background jobs, immutable action records, shared service/CLI
paths, secret-safe errors and constrained launch of a verified Rerun recording.

P08 is partially accepted. The deterministic floor-reference layer and backend/application
foundation are accepted for internal R&D use. Further web-console visual polish, workflow
ergonomics, operator guidance and production-style usability refinement are deferred.

This milestone does not establish survey-grade XYZ accuracy, as-built truth, accepted camera
connectivity or client-acceptance geometry. Public release excludes camera imagery, facility
coordinates, client material, credentials, machine-local paths and generated artifacts.
