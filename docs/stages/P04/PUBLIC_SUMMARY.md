# P04 Public Summary - Calibration and registration pilot

P04 implemented a low-touch calibration and world-registration pilot for the strongest-observed
camera. The reusable workflow covers immutable frame review, linked image/facility annotation,
multi-frame GeoCalib model comparison, explicit `T_world_from_camera` PnP/RANSAC, bounded robust
refinement, held-out checks, leave-one-out sensitivity, physical diagnostics and a complete solver
envelope.

The pilot was partially accepted. It produced a provisional coarse-registration baseline and
evidence-derived review limits, but it did not establish full XYZ accuracy. The private pilot used
one annotated landmark frame, two held-out points and a facility reference whose horizontal
uncertainty remains unknown. Those limitations prohibit sub-decimetre claims, tight camera fusion
or treating the result as a universal policy.

An equal-camera intrinsic study retained robust fleet profiles as nominal priors while rejecting a
single forced profile for every camera. Camera-specific and fleet candidates must remain comparable
until each camera has held-out pose and physical validation. The final cross-camera selection,
regularization, aggregation and rollback policy is deliberately deferred until that evidence is
available.

Client imagery, floor-plan derivatives, landmark coordinates, camera transforms, endpoint values,
model weights and private run manifests are excluded. The included synthetic tests cover success,
rejection and failure behavior for annotation, intrinsic pooling, projection, pose recovery,
outliers, degeneracy, bounds and envelope construction.
