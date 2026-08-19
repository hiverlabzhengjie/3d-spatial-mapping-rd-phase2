# P06 public milestone summary

P06 adds a typed, evidence-bounded reconstruction-mode evaluation workflow for the exact
DA3 Nested Giant-Large 1.1 model. It defines an allow-listed case matrix, explicit camera-transform
directions, immutable artifact identities, raw-field validation, deterministic repeatability,
masked depth/confidence comparisons and synthetic cross-view projection checks. A bounded direct
Camera 2/3 diagnostic was added after owner visual review identified substantial shared structure.

The reusable result is a conservative policy: retain independent single-view metric baselines and
keep pose-conditioned multi-view output diagnostic until camera registration and connectivity have
independent validation. Visible overlap was sufficient to motivate the experiment, but the current
provisional pose-conditioned pair did not outperform the three-view consistency evidence. The
workflow explicitly prohibits pose promotion, inferred connectivity, cloud fusion and facility-
frame replacement from model self-consistency.

The public milestone should include only reusable source, tests and this sanitized summary. It
must exclude client images, camera transforms, local paths, raw model outputs, facility details and
internal continuity records. Publication remains a control-tower action after stage review.
