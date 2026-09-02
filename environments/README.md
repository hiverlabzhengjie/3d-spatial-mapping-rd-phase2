# Isolated uv runtime domains

The repository deliberately has three independent dependency domains. They are separate uv
projects, not members of one uv workspace, because their NumPy and GPU-package requirements are
incompatible by design.

| Domain | Project | Lock authority | Purpose |
| --- | --- | --- | --- |
| Native console | repository root | `uv.lock` | Package, console, tests and ordinary CPU I/O |
| DA3 | `environments/da3` | `environments/da3/uv.lock` | Minimal Windows/CUDA substrate for accepted DA3 source |
| XR02 | `environments/xr02` | `environments/xr02/uv.lock` | Windows/CUDA tracking, ReID and live media stack |

Every project requires uv `0.12.7` and CPython `3.11.4`. Lockfiles are committed and managed only
through uv. Do not create a root uv workspace: a workspace would share one resolver/lock while the
native console requires NumPy 1.26 and XR02 requires NumPy 2.2.

The GPU locks describe Python runtime substrates. Accepted DA3 source/checkpoint identities,
project-wheel overlays, model files, CUDA drivers and external binaries remain separate deployment
authorities. Follow `docs/hardening/DEPENDENCY_MANAGEMENT.md`; do not install a GPU substrate over
an accepted historical environment in place.
