# P02 Public Summary - Facility-frame registration foundation

P02 added reusable typed services and a localhost workflow for registering a scanned floor plan
into a right-handed, metre-based facility frame. The workflow supports explicit scale controls,
origin and axis selection, camera mounting-reference placement, versioned state, uncertainty
fields, credential-separated endpoint bindings and credential-free exports.

The selected private pilot result was partially accepted. It used one verified scale control and
produced four provisional mounting-reference records, but a single control cannot quantify local
scan distortion, anisotropy or scale disagreement. Horizontal uncertainty therefore remains
unknown. These records are mounting-location priors only: they are not optical centres, calibrated
camera poses, overlap evidence, geometry or an XYZ-accuracy claim.

Client floor plans, annotations, coordinates, endpoint values and private exports are intentionally
excluded from this public snapshot. The included tests use synthetic fixtures and cover coordinate
transforms, revisioning, validation failures, secret separation and export behavior.
