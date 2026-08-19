# P05 - Fixed-Centre Camera Orientation and Connectivity Evidence

P05 implemented a reusable camera-registration workflow for fixed cameras whose optical centres
are supplied by mounting or survey evidence. The operational solver estimates orientation only:
it converts image observations and facility landmarks into bearing directions, enumerates every
three-of-four correspondence subset, solves rotation with Wahba/SVD, scores all four observations
and robustly refines only the three rotation parameters. Translation remains fixed.

The implementation includes:

- immutable per-camera intrinsic candidates and explicit rollback provenance;
- deterministic solve-set selection and rejection for weak conditioning, ambiguity, failed
  consensus, cheirality and excessive pixel-perturbation sensitivity;
- a sealed two-point held-out validation interface that cannot influence solving or selection;
- typed, versioned camera and connectivity records with explicit provisional and rejected states;
- a separately labelled consumed-observation fallback for provisional experiment initialization;
- synthetic tests covering exact recovery, one- and two-outlier behavior, rejection paths,
  fixed-centre invariance, sensitivity and validation separation.

P05 is partially accepted. The private pilot did not produce an accepted camera pose or an
accepted connectivity edge. One strict result remains provisional and the other strict results
are rejected; the fallback hypotheses are initialization evidence only. Site observations,
facility coordinates, camera transforms, residual overlays and private manifests are deliberately
not published.

The policy checkpoint after P05 retains the fixed-centre four-point method as the only current
acceptance path. Diagnostic 6-DoF PnP and consumed-observation estimates cannot establish camera
authority, connectivity, scale, fusion or geometry acceptance. Subsequent model experiments must
preserve those evidence labels and rollback boundaries.
