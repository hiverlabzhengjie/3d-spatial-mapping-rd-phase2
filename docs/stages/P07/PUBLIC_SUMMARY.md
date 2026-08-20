# P07 Public Summary - World-space geometry and controlled fusion

P07 established reusable, tested machinery for converting metric depth into camera-space point
clouds, transforming derived copies into a shared working frame, applying traceable filters and
combining selected patches while preserving source-camera membership.

The stage also added deterministic camera-removal behavior and a Rerun inspection artifact with
four calibrated camera frustums, image planes, orientation axes and labels. Raw inputs, derived
geometry versions and rollback identities remain separate and traceable.

For this internal R&D iteration, owner review selected an all-four pose-conditioned DA3 result as
the current working facility geometry. The camera transforms remained fixed, and the selected
cloud retains per-camera provenance and removable inputs. An earlier single-view aggregate remains
available as a reproducible rollback.

This milestone is accepted as a demonstrable working-geometry workflow, not as survey-grade XYZ
accuracy, verified as-built truth or client-acceptance geometry. Independent accuracy and
robustness validation remains future work.

Public release scope excludes camera imagery, facility coordinates, transforms, client material,
credentials, machine-local paths and generated geometry artifacts.
