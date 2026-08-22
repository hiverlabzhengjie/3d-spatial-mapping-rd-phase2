# P08 Integrated Workflow Console

## Purpose

The console is a localhost, human-facing path through facility setup, capture, four-camera
calibration correspondence, static reconstruction, floor refinement and final review. Internal
stage codes, decision codes, authority tables and operator records remain in project evidence and
are not presented as user workflow.

Reusable domain logic remains in existing services. The console mounts standalone facility,
capture and camera-correspondence tools through adapters; those standalone launchers remain
available. Camera IDs, count, paths and current result identities are configuration, not business
logic.

## Configuration

Start from these credential-free examples:

- `configs/p08_scene_workspace.example.json` for the scene and selected artifacts;
- `configs/p08_operator_workflow.example.json` for the exact camera-input manifest, selected
  geometry/floor previews and initial approval state;
- `configs/p08_frozen_source_contract.example.json` for frozen floor inputs; and
- `configs/p08_floor_config.example.json` for the authoritative plane boundary.

Keep actual paths and facility identities in ignored `configs/*.local.json` files. A scene is
immutable once created:

```powershell
$env:PYTHONPATH = '<repository>\src'
<python> scripts/p08_workflow_cli.py `
  --workspace-dir <new-workflow-workspace> `
  create-scene --scene-config <scene.local.json>
```

## Launch

```powershell
$env:PYTHONPATH = '<repository>\src'
<python> scripts/run_p08_workflow_console.py `
  --workspace-dir <workflow-workspace> `
  --operator-profile <operator-workflow.local.json> `
  --floor-contract <source-contract.local.json> `
  --floor-config <floor-config.local.json> `
  --floor-output-root <immutable-run-root> `
  --workflow-python <pinned-python> `
  --repository-root <repository> `
  --p06-run-directory <selected-input-run> `
  --da3-source-directory <pinned-da3-source> `
  --da3-checkpoint-directory <pinned-checkpoint> `
  --d041-manifest <rollback-manifest> `
  --reconstruction-output-root <immutable-run-root> `
  --expected-geometry-sha256 <selected-geometry-sha256> `
  --viewer <rerun.exe> `
  --allowed-artifact-root <artifact-root> `
  --p02-workspace <facility-workspace> `
  --calibration-workspace camera-1=<camera-1-workspace> `
  --calibration-workspace camera-2=<camera-2-workspace> `
  --enable-live-p03 --secret-file <ignored-secret-file> `
  --host 127.0.0.1 --port 8088
```

Repeat `--calibration-workspace CAMERA_ID=PATH` for the configured roster. Only `localhost`,
`127.0.0.1` and `::1` are accepted bind targets.

## Workflow gates

1. Every enabled camera must have a hash-verified input record with native intrinsics and a finite
   `T_world_from_camera`.
2. Static reconstruction is enabled only when every camera and pinned runtime dependency is ready.
3. Geometry must be opened in Rerun before it can be approved.
4. Floor generation is enabled only after geometry approval.
5. A completed floor run must be verified and its preview built/opened before final approval.

Floor generation never edits the selected geometry. It writes a new run containing every original
point/RGB unchanged plus the separate four-vertex/two-triangle mathematical floor plane at `Z=0`.

## Repeated runs and retained artifacts

The immutable scene and external run artifacts are separate from mutable operator state. The
workspace's atomic `operator-state.json` records only current geometry/floor pointers, the exact
active geometry source identity and preview/approval state, so the operator can resume after a
console restart.

**Start a fresh run** archives the prior state and terminal activity under
`operator-state-archive/`, then clears active pointers. It does not delete or overwrite capture,
geometry, floor or Rerun artifacts, and it retains the latest approved final result as a read-only
fallback. The action is refused while a long job is active. External artifact retention or cleanup
is deliberately separate from an operator reset and is handled only on **Scene history & storage**.

Every successful reconstruction registers its own combined-geometry and Rerun hashes as the active
result. Floor generation binds to that approved geometry hash. Rebuilding an already complete floor
preview verifies and reuses the existing RRD/manifest pair; partial output fails explicitly.

## Scene history and storage

Each scene workspace owns a scene-bound `artifact-catalog.sqlite3`. It is a mutable index and
selection/audit store over immutable external files; it is not a replacement for the artifacts.
The **Scene history & storage** page projects the internal file milestones into the same six human
workflow steps used by the rest of the console. It separates the current workflow, past versions
and storage cleanup. Technical filenames, paths and hashes remain under **Manage**.

The page allows the operator to:

- see the current six-step version chain and filter past versions by workflow step;
- check a past file against its recorded SHA-256;
- select a prior available version after reviewing the exact downstream reset impact;
- archive a non-current version from normal use without deleting or moving its file; and
- restore an archived version and inspect the scene-local version event history; or
- permanently delete one exact non-current, dependency-free file after reviewing its impact and
  pressing the final delete button; or
- for reconstruction, floor and final history, browse versions older than the newest five, filter
  them by timestamp and delete a dependency-safe selected batch.

Selecting an upstream version clears downstream selections and review approvals transactionally.
Static reconstruction is blocked until all required input milestones have a current version. A
selected reconstruction input manifest reconfigures both the camera-summary and DA3 adapters to
that exact manifest/run directory. A selected combined point cloud, geometry Rerun, floor run or
final Rerun updates the corresponding workflow pointer; generated artifacts are catalogued on job
completion. Current versions and versions used by a current downstream result cannot be archived.
Permanent deletion additionally requires one unshared regular file under a managed scene root and
a final matching SHA-256. It never removes a directory. The payload cannot be restored from the
  console; its path, identity, byte count and deletion event remain in the backend audit and it is
  omitted from normal operator version lists.

## CLI parity

The CLI constructs the same service and supports `status`, `reconstruct`, `floor`,
`floor-preview`, `launch-rerun`, `approve`, `artifacts`, `artifact-impact`, `artifact-select`,
`artifact-verify`, `artifact-archive`, `artifact-restore`, `artifact-delete-impact` and
`artifact-delete`, `artifact-delete-batch-impact` and `artifact-delete-batch`. Every action ID and
job ID must be new. Selection requires `--confirm-impacts` when the impact report lists downstream
resets. Permanent deletion uses the file-specific token returned by the corresponding impact
command; operators do not type a confirmation phrase in the web console.
Examples:

```powershell
<python> scripts/p08_workflow_cli.py <shared-options> `
  reconstruct --job-id <new-id> --wait-seconds 600

<python> scripts/p08_workflow_cli.py <shared-options> `
  floor --job-id <new-id>

<python> scripts/p08_workflow_cli.py <shared-options> `
  floor-preview --job-id <new-preview-id> --floor-job-id <completed-floor-id>

<python> scripts/p08_workflow_cli.py <shared-options> `
  launch-rerun --action-id <new-id> --artifact-id <selected-recording-id>
```

## Safety and recovery

- Jobs are capacity-bounded and expose queued, running, complete, failed and cancelled states.
- Long-process output is redirected to a temporary file to avoid pipe backpressure; only a bounded
  failure tail is returned, with credential/token redaction by the job manager.
- Rerun launch requires a selected `.rrd` beneath an allowed root, verifies its SHA-256 and invokes
  only the configured `rerun.exe --port 0 <recording>` command with `shell=False`.
- A stopped console deletes nothing. Restart it against the immutable workspace or create a new
  workspace when selected identities change.
- Incomplete external runs remain diagnostic evidence until explicitly reviewed; no automatic
  cleanup or overwrite occurs.
