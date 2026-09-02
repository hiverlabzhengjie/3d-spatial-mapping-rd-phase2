const state = { pages: [], status: null, artifactCatalog: null, selectedCamera: null, artifactSection: "all", deleteImpact: null, warningId: null, sceneRegistry: [] };

function selectedSceneId() {
  const match = window.location.pathname.match(/^\/scenes\/([^/]+)(?:\/|$)/);
  return match ? match[1] : null;
}

function sceneBase() {
  const sceneId = selectedSceneId();
  return sceneId ? `/scenes/${encodeURIComponent(sceneId)}` : "";
}

function scenePage(pageId) { return `${sceneBase()}/pages/${pageId}`; }

function selectedPageId() {
  const match = window.location.pathname.match(/(?:^|\/)pages\/([^/]+)$/);
  return match ? match[1] : "setup";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);
}

function humanState(value) {
  return ({
    complete: "Complete", approved: "Approved", ready: "Ready",
    ready_for_review: "Ready for review", attention: "Needs attention",
    blocked: "Blocked", pending: "Not started", available: "Verified",
    not_provisioned: "Needs setup", unhealthy: "Needs attention",
    archived: "Archived", missing: "Missing", corrupt: "Changed", deleted: "Deleted permanently"
  })[value] || String(value || "Unknown").replaceAll("_", " ");
}

function uniqueId(prefix) {
  return `${prefix}-${new Date().toISOString().replace(/[-:.TZ]/g, "").toLowerCase()}`;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}

function workflowCapability(capabilityId) {
  const capabilities = asObject(state.status?.operator?.workflow_capabilities);
  return asObject(capabilities?.[capabilityId]);
}

function managedSceneInput(inputId) {
  return asObject(asObject(state.status?.scene_inputs)?.[inputId]);
}

function sceneNotProvisioned(title, capabilityId, previousPageId, previousLabel) {
  const capability = workflowCapability(capabilityId);
  const capture = managedSceneInput("capture");
  const waitingForCapture = capabilityId === "calibration" && capture && !capture.ready;
  const message = waitingForCapture
    ? asArray(capture.issues)[0] || "Select one complete camera bundle first."
    : capability?.message || "This workflow step is not set up for this scene yet.";
  const heading = waitingForCapture
    ? `${title} is waiting for a complete capture`
    : `${title} is not set up for this scene yet`;
  return `<article class="card hero wide"><p class="eyebrow">Scene-specific setup</p><h2>${escapeHtml(heading)}</h2><p>${escapeHtml(message)}</p><a class="button-link" href="${scenePage(previousPageId)}">${escapeHtml(previousLabel)}</a></article>`;
}

async function api(url, options = {}) {
  const sceneId = selectedSceneId();
  const target = sceneId && url.startsWith("/api/") && url !== "/api/scenes"
    ? `/api/scenes/${encodeURIComponent(sceneId)}${url.slice(4)}`
    : url;
  const response = await fetch(target, { cache: "no-store", ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  return payload;
}

function notify(message, error = false) {
  const element = document.getElementById("notice");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.remove("hidden");
}

function renderNavigation() {
  const selected = selectedPageId();
  const updatesUnlocked = Boolean(state.status?.scene_updates?.unlocked_at_utc);
  document.getElementById("navigation").innerHTML = state.pages.map(page => {
    const locked = page.page_id === "updates" && !updatesUnlocked;
    const heading = page.page_id === "live" ? '<span class="nav-section-label">Operations</span>' : "";
    return `${heading}<a class="${page.page_id === selected ? "active" : ""} ${locked ? "locked" : ""}" href="${locked ? scenePage("results") : scenePage(page.page_id)}" ${locked ? 'aria-disabled="true" title="Available after the first final approval"' : ""}>${escapeHtml(page.title)}${locked ? " · locked" : ""}</a>`;
  }).join("");
  const page = state.pages.find(item => item.page_id === selected) || state.pages[0];
  document.getElementById("page-title").textContent = page?.title || "Workflow";
}

function renderSteps() {
  const captureToolAvailable = Boolean(state.pages.find(page => page.page_id === "capture")?.tool_url);
  const capabilityByStep = {
    setup: "facility", capture: "capture", calibration: "calibration",
    reconstruction: "reconstruction", floor: "floor", results: "final_review"
  };
  document.getElementById("step-strip").innerHTML = state.status.operator.steps.map((step, index) => {
    // A managed scene can offer an isolated Capture tool before the later calibration and
    // reconstruction adapters are provisioned.  Show that as an actionable setup task rather
    // than a false navigation block; the backend phase and scientific gates remain unchanged.
    const capability = workflowCapability(capabilityByStep[step.step_id]);
    const displayState = capability?.state === "not_provisioned" ? "blocked" : (
      step.step_id === "capture" && captureToolAvailable && step.state === "blocked"
    ) ? "attention" : step.state;
    return `
    <a href="${scenePage(escapeHtml(step.step_id))}" class="step ${escapeHtml(displayState)}">
      <span>${index + 1}</span><div><strong>${escapeHtml(step.title)}</strong><small>${escapeHtml(humanState(displayState))}</small></div>
    </a>`;
  }).join("");
}

function statusBadge(value) {
  return `<span class="state ${escapeHtml(value)}">${escapeHtml(humanState(value))}</span>`;
}

function projectOverview() {
  const workflow = state.status.operator;
  const calibrationCapability = workflowCapability("calibration");
  const facilityInput = managedSceneInput("facility");
  const captureInput = managedSceneInput("capture");
  const completed = workflow.steps.filter(item => ["complete", "approved"].includes(item.state)).length;
  const retainedCurrent = asArray(state.artifactCatalog?.workflow_sections)
    .reduce((count, section) => count + asArray(section.current_items).length, 0);
  const guidance = facilityInput && !facilityInput.ready
    ? asArray(facilityInput.issues)[0] || "Complete Facility & cameras."
    : captureInput && !captureInput.ready
    ? asArray(captureInput.issues)[0] || "Select one complete camera bundle."
    : calibrationCapability?.state === "not_provisioned"
    ? "Facility and Capture are complete. Calibration and later processing still need scene-specific setup."
    : workflow.inputs_ready
    ? "Camera calibration and world poses are ready for all configured cameras. Continue from the next available step."
    : "Begin with Facility & cameras. The console will show each next step when this scene is ready for it.";
  const hasActivity = Boolean(workflow.session.has_activity);
  const historyContext = retainedCurrent
    ? `This current run is independent from ${retainedCurrent} retained current artifact ${retainedCurrent === 1 ? "version" : "versions"}. Those verified files remain available in Scene History & Storage until you deliberately select or regenerate them.`
    : "This current run has no retained current artifact versions yet.";
  return `
    <article class="card hero wide"><p class="eyebrow">Current project</p><h2>${escapeHtml(state.status.display_name)}</h2>
      <p>${escapeHtml(guidance)}</p>
      <p class="run-context"><strong>Current run:</strong> ${escapeHtml(historyContext)} <a href="${scenePage("artifacts")}">Review scene history</a></p>
      <div class="metric-grid"><div class="metric"><strong>${state.status.camera_roster.length}</strong><span>Cameras</span></div>
      <div class="metric"><strong>${completed}/${workflow.steps.length}</strong><span>Current-run steps</span></div>
      <div class="metric"><strong>${workflow.inputs_ready ? "Ready" : "Check"}</strong><span>Reconstruction inputs</span></div></div>
      <div class="session-actions"><div><h3>Start over safely</h3><p>${hasActivity ? "Archive this run and clear its processing activity without deleting captures or generated artifacts." : "This scene has no processing activity to clear yet."}</p></div><button id="fresh-session" class="secondary" ${workflow.session.can_start_fresh && hasActivity ? "" : "disabled"}>Start a fresh run</button></div>
    </article>
    ${cameraRoster()}`;
}

function formatBytes(value) {
  if (value === null || value === undefined) return "File unavailable";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function artifactVersionsPage() {
  const catalog = state.artifactCatalog;
  if (!catalog) return `<article class="card wide"><h2>Loading scene history…</h2></article>`;
  const milestoneByKey = Object.fromEntries(catalog.milestones.map(item => [item.milestone_key, item]));
  const sections = catalog.workflow_sections || [];
  const currentCards = sections.map((section, index) => {
    const items = section.current_items || [];
    const currentRows = items.length ? items.map(version => {
      const milestone = milestoneByKey[version.milestone_key];
      return `<div class="current-item"><div><strong>${escapeHtml(milestone?.title || section.title)}</strong><small>${escapeHtml(new Date(version.created_at).toLocaleString())}</small></div>${statusBadge(version.lifecycle)}</div>`;
    }).join("") : `<p class="empty-inline">No current version selected</p>`;
    return `<article class="current-stage ${items.length ? "has-current" : "needs-current"}"><span class="stage-number">${index + 1}</span><div class="current-stage-body"><h3>${escapeHtml(section.title)}</h3><p>${escapeHtml(section.description)}</p><div class="current-items">${currentRows}</div></div></article>`;
  }).join("");
  const filters = [`<button type="button" class="history-filter ${state.artifactSection === "all" ? "active" : ""}" data-section="all">All steps</button>`, ...sections.map(section => `<button type="button" class="history-filter ${state.artifactSection === section.section_key ? "active" : ""}" data-section="${escapeHtml(section.section_key)}">${escapeHtml(section.title)} <span>${section.past_count}</span></button>`)].join("");
  const visibleSections = state.artifactSection === "all" ? sections : sections.filter(section => section.section_key === state.artifactSection);
  const pastGroups = visibleSections.map(section => {
    const versions = (section.past_items || []).filter(version => version.lifecycle !== "deleted" && version.metadata.selectable !== false);
    if (!versions.length) return "";
    const highVolume = ["reconstruction", "floor", "final"].includes(section.section_key);
    const recent = highVolume ? versions.slice(0, 5) : versions;
    const older = highVolume ? versions.slice(5) : [];
    const recentRows = recent.map(version => renderPastVersion(version, milestoneByKey[version.milestone_key]?.title || section.title)).join("");
    const olderRows = older.map(version => renderPastVersion(version, milestoneByKey[version.milestone_key]?.title || section.title, section.section_key)).join("");
    const olderHistory = older.length ? `<details class="older-history" data-section="${escapeHtml(section.section_key)}"><summary>Browse ${older.length} older ${older.length === 1 ? "file" : "files"}</summary><div class="older-history-body"><div class="older-toolbar"><label>From<input type="datetime-local" id="history-from-${escapeHtml(section.section_key)}"></label><label>To<input type="datetime-local" id="history-to-${escapeHtml(section.section_key)}"></label><button class="secondary apply-history-filter" data-section="${escapeHtml(section.section_key)}">Apply dates</button><button class="secondary clear-history-filter" data-section="${escapeHtml(section.section_key)}">Clear</button></div><div class="batch-actions"><button class="secondary select-visible-history" data-section="${escapeHtml(section.section_key)}">Select visible</button><button class="danger batch-delete-history" data-section="${escapeHtml(section.section_key)}" disabled>Delete selected files…</button></div><div class="older-list" id="older-list-${escapeHtml(section.section_key)}">${olderRows}</div></div></details>` : "";
    return `<section class="past-stage"><div class="past-stage-heading"><h3>${escapeHtml(section.title)}</h3><span>${versions.length} past ${versions.length === 1 ? "version" : "versions"}${highVolume && versions.length > 5 ? " · 5 newest shown" : ""}</span></div><div class="past-list">${recentRows}</div>${olderHistory}</section>`;
  }).filter(Boolean).join("");
  const storage = catalog.storage || {};
  const recentEvents = catalog.events.slice(0, 6).map(event => `<div class="history-row"><strong>${escapeHtml(humanState(event.action))}</strong><span>${escapeHtml(new Date(event.occurred_at).toLocaleString())}</span></div>`).join("");
  return `<section class="history-page">
    <article class="history-intro"><p class="eyebrow">This scene only</p><h2>Scene history & storage</h2><p>Keep one clear current workflow, return to an older version when needed, and remove old files only after checking their impact.</p>
      <ol class="how-it-works"><li><strong>Current workflow</strong><span>The versions used by the console now.</span></li><li><strong>Past versions</strong><span>Older work stays available without overwriting the current result.</span></li><li><strong>Storage cleanup</strong><span>Permanent deletion shows the exact file and dependencies first.</span></li></ol>
    </article>
    <section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Current workflow</p><h2>What the console is using now</h2></div><span class="state approved">${storage.current_version_count || 0} current versions</span></div><div class="current-stage-grid">${currentCards}</div></section>
    ${cameraPolicyHistory()}
    ${calibrationHistory()}
    ${liveOperationsHistory()}
    <section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Past versions</p><h2>Return to earlier work</h2><p>Select a workflow step, then choose <strong>Make current</strong>. If later work must be reset, you will see that before anything changes.</p></div></div><div class="history-filters" role="tablist" aria-label="Filter past versions by workflow step">${filters}</div>${pastGroups || `<div class="empty-history"><h3>No past versions in this view</h3><p>Older versions will appear here after another run or selection is created.</p></div>`}</section>
    <section class="history-section cleanup-section"><div><p class="eyebrow">Storage cleanup</p><h2>Remove old files carefully</h2><p>Open <strong>Manage</strong> beside a past version to check, archive, or permanently delete it. Current files, accepted inputs, authority records, rollback records and files needed by another retained result are protected.</p></div><div class="storage-metrics"><div><strong>${storage.past_version_count || 0}</strong><span>Past versions</span></div><div><strong>${escapeHtml(formatBytes(storage.past_retained_byte_count || 0))}</strong><span>Past files retained</span></div><div><strong>${storage.protected_version_count || 0}</strong><span>Protected records</span></div><div><strong>${storage.deleted_version_count || 0}</strong><span>Deleted records</span></div></div><div class="danger-note"><strong>Permanent means permanent.</strong> Only the exact displayed file is removed. The console keeps its name, hash and deletion record, but cannot restore the file.</div></section>
    <details class="recent-changes"><summary>Recent history actions</summary><div class="history-list">${recentEvents || `<p class="muted">No management actions recorded yet.</p>`}</div></details>
    ${deletionDialog()}
  </section>`;
}

function liveOperationsHistory() {
  if (workflowCapability("live_operations")?.state === "not_provisioned") {
    return `<section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Live operations</p><h2>Operational run history</h2></div></div><p class="muted">Live operations are not set up for this scene yet.</p></section>`;
  }
  const worker = asObject(state.status?.live_operations?.worker);
  if (!worker) return `<section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Live operations</p><h2>Operational run history</h2></div></div><p class="muted">XR02 worker history is unavailable in this launch.</p></section>`;
  const runs = [...asArray(worker.saved_recordings), ...asArray(worker.recent_live_runs)].sort((a, b) => Date.parse(b.started_at_utc) - Date.parse(a.started_at_utc));
  const rows = runs.map(run => `<div class="history-row"><div><strong>${escapeHtml(run.label || (run.mode === "live" ? "Live Service" : "Replayable Recording"))}</strong><small>${escapeHtml(new Date(run.started_at_utc).toLocaleString())} · ${escapeHtml(formatBytes(run.byte_count))}</small><small>Scene ${escapeHtml((run.scene_binding_sha256 || "legacy-unscoped").slice(0, 12))} · ${escapeHtml(humanState(run.state))}</small></div></div>`).join("");
  return `<section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Live operations</p><h2>Scene-bound operating history</h2><p>Operational recordings and compact Live telemetry remain separate from selectable geometry authority.</p></div><span class="state ${worker.active ? "running" : "approved"}">${worker.active ? "Active" : `${runs.length} retained`}</span></div><div class="history-list">${rows || '<p class="muted">No retained Live or Recording runs yet.</p>'}</div></section>`;
}

function renderPastVersion(version, milestoneTitle, batchSection = null) {
  if (version.lifecycle === "deleted") return "";
  const canSelect = version.lifecycle === "available";
  const archived = version.lifecycle === "archived";
  const protectedRecord = Boolean(version.retention);
  const protectionBadge = protectedRecord ? `<span class="state approved">Protected</span>` : "";
  const batchSelector = batchSection ? `<label class="batch-select"><input type="checkbox" class="batch-version-checkbox" data-section="${escapeHtml(batchSection)}" data-artifact="${escapeHtml(version.artifact_id)}" ${protectedRecord ? "disabled" : ""}><span>Select</span></label>` : "";
  return `<article class="past-version ${archived ? "archived" : ""} ${batchSection ? "batch-version" : ""}" data-created="${escapeHtml(version.created_at)}">${batchSelector}<div class="past-version-main"><div class="past-version-title"><div><span class="version-kind">${escapeHtml(milestoneTitle)}</span><strong>${escapeHtml(new Date(version.created_at).toLocaleString())}</strong></div>${protectionBadge}${statusBadge(version.lifecycle)}</div><small>${escapeHtml(formatBytes(version.byte_count))}</small></div><div class="past-version-actions"><button class="select-version" data-artifact="${escapeHtml(version.artifact_id)}" ${canSelect ? "" : "disabled"}>Make current</button><details class="manage-version"><summary>Manage</summary><div class="manage-menu"><button class="secondary verify-version" data-artifact="${escapeHtml(version.artifact_id)}">Check file</button><button class="secondary archive-version" data-artifact="${escapeHtml(version.artifact_id)}" data-archived="${archived ? "false" : "true"}">${archived ? "Restore from archive" : "Archive from normal use"}</button><button class="danger delete-version" data-artifact="${escapeHtml(version.artifact_id)}" ${protectedRecord ? "disabled title=\"Protected authority record\"" : ""}>Delete permanently…</button>${protectedRecord ? `<p class="muted">${escapeHtml(version.retention.reason)}</p>` : ""}<details class="file-details"><summary>Technical file details</summary><dl class="artifact-details"><dt>File</dt><dd>${escapeHtml(version.display_name)}</dd><dt>Location</dt><dd>${escapeHtml(version.path)}</dd><dt>SHA-256</dt><dd>${escapeHtml(version.sha256)}</dd></dl></details></div></details></div></article>`;
}

function deletionDialog() {
  return `<dialog id="delete-dialog" class="delete-dialog"><form method="dialog"><button class="dialog-close" value="cancel" aria-label="Close">×</button></form><div id="delete-dialog-content"><h2>Checking deletion impact…</h2></div></dialog>`;
}

function cameraRoster() {
  return `<article class="card wide"><h2>Configured cameras</h2><div class="camera-grid">${asArray(state.status?.camera_roster).map(camera => `
    <div class="camera-tile"><span class="camera-icon">●</span><div><strong>${escapeHtml(camera.display_name)}</strong><small>${camera.enabled ? "Included in this scene" : "Disabled"}</small></div></div>`).join("")}</div></article>`;
}

function currentCameraPolicy() {
  const status = asObject(state.status?.camera_policy);
  return asObject(status?.active_policy) || asObject(status?.proposal);
}

function cameraPolicyCompatibilityNotice() {
  return `<article class="card wide policy-card"><div class="warning"><strong>Camera-policy controls need a console restart.</strong> This page is connected to an older in-memory backend while newer interface files are present. Existing scene data and artifact history are unchanged; restart the integrated console to load the matching camera-policy service.</div></article>`;
}

function lensGroupEditor() {
  const status = asObject(state.status?.camera_policy);
  const policy = currentCameraPolicy();
  if (!status || !policy) return cameraPolicyCompatibilityNotice();
  const membership = {};
  asArray(policy.intrinsic_groups).forEach(group => asArray(group.camera_ids).forEach(cameraId => { membership[cameraId] = group; }));
  const cameraIds = asArray(policy.camera_ids).length
    ? asArray(policy.camera_ids)
    : asArray(state.status?.camera_roster).filter(camera => camera.enabled).map(camera => camera.camera_id);
  const rows = cameraIds.map(cameraId => {
    const camera = asArray(state.status?.camera_roster).find(item => item.camera_id === cameraId);
    const group = membership[cameraId] || {};
    return `<div class="policy-row lens-policy-row" data-camera-id="${escapeHtml(cameraId)}"><div><strong>${escapeHtml(camera?.display_name || cameraId)}</strong><small>${escapeHtml(cameraId)}</small></div><label><span>Lens group</span><input class="lens-group-id" value="${escapeHtml(group.group_id || "")}" placeholder="e.g. lens-a"></label><label><span>Lens model</span><input class="lens-model" value="${escapeHtml(group.lens_model || "")}" placeholder="Manufacturer / model"></label></div>`;
  }).join("");
  return `<article class="card wide policy-card"><div class="section-heading"><div><p class="eyebrow">Intrinsic eligibility</p><h2>Group cameras by lens model</h2><p>Cameras in a group are eligible for comparison under the existing intrinsic policy. Grouping never forces a shared calibration.</p></div>${status.active_policy ? statusBadge(policy.lens_complete ? "complete" : "attention") : statusBadge("pending")}</div><div class="policy-list">${rows}</div><div class="policy-actions"><button id="save-lens-policy">Save lens groups</button><small>Revision ${status.active_revision || "not configured"} · ${policy.lens_complete ? "all enabled cameras assigned" : "assignment incomplete"}</small></div></article>`;
}

function overlapPairMarkup(cameraIdA, cameraIdB, names) {
  return `<div class="policy-row overlap-policy-row" data-camera-a="${escapeHtml(cameraIdA)}" data-camera-b="${escapeHtml(cameraIdB)}"><div><strong>${escapeHtml(names[cameraIdA] || cameraIdA)} ↔ ${escapeHtml(names[cameraIdB] || cameraIdB)}</strong><small>${escapeHtml(cameraIdA)} · ${escapeHtml(cameraIdB)}</small></div><button type="button" class="secondary remove-overlap-pair">Remove</button></div>`;
}

function overlapEditor() {
  const status = asObject(state.status?.camera_policy);
  const policy = currentCameraPolicy();
  if (!status || !policy) return cameraPolicyCompatibilityNotice();
  const names = Object.fromEntries(asArray(state.status?.camera_roster).map(camera => [camera.camera_id, camera.display_name]));
  const cameraIds = asArray(policy.camera_ids);
  const options = cameraIds.map(cameraId => `<option value="${escapeHtml(cameraId)}">${escapeHtml(names[cameraId] || cameraId)}</option>`).join("");
  const overlapReviews = asArray(policy.overlap_pair_reviews).filter(review => review.verdict === "overlap");
  const rows = overlapReviews.map(review => overlapPairMarkup(review.camera_id_a, review.camera_id_b, names)).join("");
  const totalPairs = cameraIds.length * (cameraIds.length - 1) / 2;
  const editor = totalPairs
    ? `<div class="overlap-composer"><label><span>First camera</span><select id="overlap-camera-a">${options}</select></label><label><span>Second camera</span><select id="overlap-camera-b">${options}</select></label><button type="button" id="add-overlap-pair">Add overlapping pair</button></div><div class="policy-list" id="overlap-pair-list">${rows || '<p class="muted overlap-empty">No overlapping pairs declared.</p>'}</div>`
    : '<p class="muted">A one-camera scene has no overlap pairs to declare.</p>';
  return `<article class="card wide policy-card"><div class="section-heading"><div><p class="eyebrow">XR02 relationship review</p><h2>Declare only camera pairs whose views overlap</h2><p>Pairs not listed default to no overlap. Each declared pair remains independent and is used only to deduplicate simultaneous genuine-overlap views in XR02; this does not affect DA3 static reconstruction or handoff topology.</p></div>${status.active_policy ? statusBadge(policy.overlap_complete ? "complete" : "attention") : statusBadge("pending")}</div>${editor}<div class="policy-actions"><button id="save-overlap-policy">Save overlap declarations</button><small id="overlap-summary">${overlapReviews.length} overlapping · ${totalPairs - overlapReviews.length} default no overlap · revision ${status.active_revision || "not configured"}</small></div></article>`;
}

function updateOverlapSummary() {
  const policy = currentCameraPolicy();
  const cameraCount = asArray(policy?.camera_ids).length;
  const totalPairs = cameraCount * (cameraCount - 1) / 2;
  const overlapCount = document.querySelectorAll(".overlap-policy-row").length;
  const summary = document.getElementById("overlap-summary");
  if (summary) summary.textContent = `${overlapCount} overlapping · ${totalPairs - overlapCount} default no overlap · revision ${asObject(state.status?.camera_policy).active_revision || "not configured"}`;
}

function bindOverlapRemoval(button) {
  button.addEventListener("click", () => {
    button.closest(".overlap-policy-row")?.remove();
    const list = document.getElementById("overlap-pair-list");
    if (list && !list.querySelector(".overlap-policy-row")) list.innerHTML = '<p class="muted overlap-empty">No overlapping pairs declared.</p>';
    updateOverlapSummary();
  });
}

function addOverlapPair() {
  const policy = currentCameraPolicy();
  const cameraIds = asArray(policy?.camera_ids);
  const cameraIdA = document.getElementById("overlap-camera-a")?.value;
  const cameraIdB = document.getElementById("overlap-camera-b")?.value;
  if (!cameraIdA || !cameraIdB || cameraIdA === cameraIdB) {
    notify("Choose two different cameras.", true);
    return;
  }
  const leftIndex = cameraIds.indexOf(cameraIdA);
  const rightIndex = cameraIds.indexOf(cameraIdB);
  const [left, right] = leftIndex < rightIndex ? [cameraIdA, cameraIdB] : [cameraIdB, cameraIdA];
  const duplicate = [...document.querySelectorAll(".overlap-policy-row")].some(row => row.dataset.cameraA === left && row.dataset.cameraB === right);
  if (duplicate) {
    notify("That overlapping pair is already declared.", true);
    return;
  }
  const names = Object.fromEntries(asArray(state.status?.camera_roster).map(camera => [camera.camera_id, camera.display_name]));
  const list = document.getElementById("overlap-pair-list");
  list?.querySelector(".overlap-empty")?.remove();
  list?.insertAdjacentHTML("beforeend", overlapPairMarkup(left, right, names));
  const addedButton = list?.lastElementChild?.querySelector(".remove-overlap-pair");
  if (addedButton) bindOverlapRemoval(addedButton);
  updateOverlapSummary();
  notify("Overlapping pair added. Save the declarations to create a policy revision.");
}

function cameraPolicyHistory() {
  const status = asObject(state.artifactCatalog?.camera_policy);
  if (!status) return "";
  const active = status.active_policy;
  const rows = asArray(status.revisions).map(item => `<article class="past-version ${item.active ? "selected" : ""}"><div class="past-version-main"><div class="past-version-title"><div><span class="version-kind">Camera policy revision ${item.revision}</span><strong>${escapeHtml(new Date(item.created_at_utc).toLocaleString())}</strong></div>${item.active ? statusBadge("approved") : ""}</div><small>${item.policy?.lens_complete ? "Lens groups complete" : "Lens groups incomplete"} · ${item.policy?.overlap_complete ? "Overlap review complete" : "Overlap review incomplete"} · ${escapeHtml(String(item.policy_sha256 || "unknown").slice(0, 12))}…</small></div><div class="past-version-actions">${item.active ? '<span class="support-note">Current</span>' : `<button class="rollback-camera-policy" data-revision="${item.revision}">Make current</button>`}</div></article>`).join("");
  return `<section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Scene policy history</p><h2>Camera relationships</h2><p>Lens eligibility and pairwise overlap use their own append-only revisions. Changing the current revision never edits scene.json or any retained artifact.</p></div>${active ? `<span class="state approved">Revision ${status.active_revision}</span>` : '<span class="state pending">Not configured</span>'}</div><div class="past-list">${rows || '<div class="empty-history"><h3>No policy revision yet</h3><p>Save lens groups from Facility & cameras to create the first revision.</p></div>'}</div></section>`;
}

function calibrationHistory() {
  if (workflowCapability("calibration")?.state === "not_provisioned") {
    return `<section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Calibration history</p><h2>Intrinsic assignments, attempts, and decisions</h2></div></div><p class="muted">Calibration is not set up for this scene yet.</p></section>`;
  }
  const history = asObject(state.artifactCatalog?.calibration_history);
  if (!history) return "";
  const rows = [
    ...asArray(history.intrinsic_batches).map(item => ({ at: item.created_at_utc, title: "Scene intrinsic assignment", detail: `${String(item.payload_sha256 || "unknown").slice(0, 12)}… · policy ${String(item.camera_policy_sha256 || "unknown").slice(0, 12)}…`, state: "available" })),
    ...asArray(history.attempts).map(item => ({ at: item.created_at_utc, title: `${item.camera_id} calibration attempt`, detail: `${item.automated_status} · ${item.selected_intrinsic_label || "no usable intrinsic/pose"} · ${String(item.payload_sha256 || "unknown").slice(0, 12)}…`, state: item.automated_status === "accepted" ? "ready_for_review" : "attention" })),
    ...asArray(history.decisions).map(item => ({ at: item.created_at_utc, title: `${item.camera_id} operator decision`, detail: `${item.decision}${item.reason ? ` · ${item.reason}` : ""}`, state: item.decision === "operator-override" ? "attention" : "approved" })),
  ].sort((left, right) => Date.parse(right.at) - Date.parse(left.at));
  return `<section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Calibration history</p><h2>Intrinsic assignments, attempts, and decisions</h2><p>This append-only scene record is separate from immutable scene.json and artifact files. Revising points makes a result stale; it does not erase the attempt.</p></div><span class="state available">${rows.length} records</span></div><div class="past-list">${rows.map(item => `<article class="past-version"><div class="past-version-main"><div class="past-version-title"><div><span class="version-kind">${escapeHtml(item.title)}</span><strong>${escapeHtml(new Date(item.at).toLocaleString())}</strong></div>${statusBadge(item.state)}</div><small>${escapeHtml(item.detail)}</small></div></article>`).join("") || '<div class="empty-history"><h3>No calibration actions yet</h3><p>Intrinsic determination will create the first history record.</p></div>'}</div></section>`;
}

function embeddedTool(page, emptyMessage) {
  if (!page.tool_url) return `<article class="card wide"><h2>Tool unavailable</h2><p>${escapeHtml(emptyMessage)}</p></article>`;
  return `<article class="card wide tool-card"><iframe class="tool-frame" src="${escapeHtml(page.tool_url)}" title="${escapeHtml(page.title)}"></iframe></article>`;
}

function calibrationPage(page) {
  const operator = asObject(state.status?.operator) || {};
  if (workflowCapability("calibration")?.state === "not_provisioned") {
    return sceneNotProvisioned("Calibration & pose", "calibration", "capture", "Return to Capture");
  }
  const roster = asArray(state.status?.camera_roster);
  const cameras = asArray(operator.cameras);
  const tools = asArray(page.calibration_tools);
  if (!state.selectedCamera) state.selectedCamera = cameras[0]?.camera_id || tools[0]?.camera_id;
  const selected = cameras.find(camera => camera.camera_id === state.selectedCamera) || cameras[0];
  const selectedTool = tools.find(tool => tool.camera_id === state.selectedCamera);
  const calibration = asObject(operator.calibration);
  if (!calibration) return `<article class="card wide"><div class="section-heading"><div><h2>Camera correspondence workspace</h2><p>Switch cameras to inspect or refine image correspondences and the resulting world pose in one place.</p></div>${selected ? statusBadge(selected.ready ? "complete" : "attention") : ""}</div><div class="camera-tabs">${roster.map(camera => `<button type="button" class="camera-tab ${camera.camera_id === state.selectedCamera ? "active" : ""}" data-camera="${escapeHtml(camera.camera_id)}">${escapeHtml(camera.display_name)}</button>`).join("")}</div>${selected ? cameraSummary(selected, false) : `<div class="warning">No camera calibration result is ready yet.</div>`}</article><article class="card wide tool-card">${selectedTool ? `<iframe class="tool-frame" src="${escapeHtml(selectedTool.tool_url)}" title="Calibration correspondence for ${escapeHtml(state.selectedCamera)}"></iframe>` : `<div class="empty"><h2>No correspondence workspace configured</h2></div>`}</article>`;
  const batch = asObject(calibration.intrinsic_batch);
  const tabs = roster.map(camera => {
    const item = cameras.find(value => value.camera_id === camera.camera_id);
    return `<button type="button" class="camera-tab ${camera.camera_id === state.selectedCamera ? "active" : ""}" data-camera="${escapeHtml(camera.camera_id)}">${escapeHtml(camera.display_name)} · ${escapeHtml(item?.ready ? "ready" : "check")}</button>`;
  }).join("");
  const assignmentCount = batch?.assignments?.length || 0;
  return `<article class="card wide intrinsic-gate"><div class="section-heading"><div><p class="eyebrow">Required first step</p><h2>1. Capture frames and determine intrinsic profiles</h2><p>Use each camera workspace below to retain at least three clear frames and approve one primary frame. Then run the pinned scene-specific GeoCalib worker. This may take a few minutes.</p></div>${statusBadge(calibration.intrinsics_ready ? "complete" : "attention")}</div><div class="button-row"><button id="determine-intrinsics" ${operator.lens_policy_ready ? "" : "disabled"}>${calibration.intrinsics_ready ? "Re-determine for current frames" : "Determine intrinsics for all cameras"}</button>${calibration.intrinsics_ready && batch ? `<span class="support-note">${assignmentCount} camera profiles assigned · ${escapeHtml(String(batch.payload_sha256 || "unknown").slice(0, 12))}…</span>` : ""}</div>${!operator.lens_policy_ready ? '<div class="warning">Complete lens grouping in Facility & cameras first.</div>' : ""}</article>
  <article class="card wide"><div class="section-heading"><div><p class="eyebrow">Per-camera loop</p><h2>2. Link points, export, and test calibration</h2><p>Collect exactly four solve points and two validation points. After <strong>Export linked points</strong>, test this camera immediately and revise until it is ready.</p></div>${selected ? statusBadge(selected.ready ? "complete" : "attention") : ""}</div><div class="camera-tabs">${tabs}</div>${selected ? calibrationCameraPanel(selected, selectedTool) : '<div class="warning">No configured camera calibration is available.</div>'}</article>
  <article id="correspondence-workspace" class="card wide tool-card">${selectedTool ? `<iframe class="tool-frame" src="${escapeHtml(selectedTool.tool_url)}" title="Calibration correspondence for ${escapeHtml(state.selectedCamera)}"></iframe>` : `<div class="empty"><h2>No correspondence workspace configured</h2><p>Mount a P04 workspace for this camera before collecting linked points.</p></div>`}</article>`;
}

function calibrationCameraPanel(camera, tool) {
  const input = camera.input || {};
  const assignment = camera.assignment;
  const attempt = camera.attempt;
  const countSummary = `${input.solve_count ?? 0}/4 solve · ${input.d034_validation_count ?? 0}/2 validation · ${input.current_export_ready ? "current export ready" : "export required"}`;
  const assignmentHtml = assignment ? `<dl class="details"><dt>Lens group</dt><dd>${escapeHtml(assignment.group_id)} · ${escapeHtml(assignment.lens_model)}</dd><dt>Initial profile</dt><dd>${escapeHtml(assignment.initial_assignment_label)}</dd><dt>Eligible profiles</dt><dd>${asArray(assignment.eligible_candidates).length}</dd></dl>` : `<p class="muted">No intrinsic assignment exists for the active camera policy.</p>`;
  let result = `<div class="calibration-result empty-result"><h3>No calibration attempt for this export</h3><p>Export the current linked points, then run the camera calibration check.</p></div>`;
  if (attempt) {
    const accepted = attempt.automated_status === "accepted";
    const validation = attempt.validation;
    const solveRmse = attempt.pose?.solve_reprojection_rmse_pixels;
    const validationRmse = validation?.descriptive_rmse_pixels;
    const diagnostics = (attempt.diagnostics || []).slice(0, 4).map(line => `<li>${escapeHtml(line)}</li>`).join("");
    const reviewAction = accepted && !camera.decision ? `<button id="review-calibration" data-attempt="${escapeHtml(attempt.payload_sha256)}">I reviewed the evidence — mark ready</button>` : "";
    const overrideAction = !accepted && attempt.can_override && !camera.decision ? `<div class="override-box"><h3>Manual review: Accept anyway</h3><p>This keeps the automated result failed and marks Static reconstruction with a warning.</p><label class="field"><span>Reason for accepting this pose</span><input id="override-reason" placeholder="Record the physical evidence and judgment"></label><label class="check-row"><input id="override-acknowledged" type="checkbox"><span>I reviewed the evidence and accept the validation risk.</span></label><button id="override-calibration" class="danger" data-attempt="${escapeHtml(attempt.payload_sha256)}">Accept anyway with warning</button></div>` : "";
    const validationErrors = asArray(validation?.individual_reprojection_errors_pixels);
    result = `<div class="calibration-result ${accepted ? "pass-result" : "fail-result"}"><div class="section-heading"><div><p class="eyebrow">Current result</p><h3>${accepted ? "Automated calibration passed" : "Automated calibration unsuccessful"}</h3></div>${statusBadge(accepted ? "ready_for_review" : "failed")}</div><dl class="details"><dt>Intrinsics used</dt><dd>${escapeHtml(attempt.selected_intrinsic_label || "No profile produced a usable pose")}</dd><dt>Solve RMSE</dt><dd>${Number.isFinite(solveRmse) ? `${Number(solveRmse).toFixed(2)} px` : "No accepted solve"}</dd><dt>Validation RMSE</dt><dd>${Number.isFinite(validationRmse) ? `${Number(validationRmse).toFixed(2)} px` : "Not evaluated"}</dd><dt>Validation points</dt><dd>${validationErrors.length ? validationErrors.map(value => `${Number(value).toFixed(2)} px`).join(" · ") : "Not evaluated"}</dd></dl><ul class="diagnostics">${diagnostics}</ul>${tool && attempt.evidence_overlay ? calibrationEvidence(tool.tool_url, attempt.evidence_overlay) : ""}<div class="button-row">${reviewAction}<button id="revise-linked-points" class="secondary">Go back and revise linked points</button></div>${overrideAction}</div>`;
  }
  return `<div class="calibration-control-grid"><div><h3>Intrinsic assignment</h3>${assignmentHtml}<h3>Linked-point readiness</h3><p class="readiness-line">${escapeHtml(countSummary)}</p>${input.reason ? `<p class="muted">${escapeHtml(input.reason)}</p>` : ""}<button id="calibrate-camera-now" ${camera.calibrate_enabled ? "" : "disabled"}>Calibrate this camera now</button></div>${result}</div>${camera.warning ? `<div class="warning"><strong>Readiness warning.</strong> ${escapeHtml(camera.warning)}</div>` : ""}`;
}

function calibrationEvidence(toolUrl, evidence) {
  const circles = (points, className) => (points || []).map(point => `<circle class="${className}" cx="${Number(point[0])}" cy="${Number(point[1])}" r="10"/>`).join("");
  const frameUrl = `${toolUrl}api/frames/${encodeURIComponent(evidence.frame_id)}/image`;
  return `<figure class="evidence-overlay"><div><img src="${escapeHtml(frameUrl)}" alt="Calibration evidence frame"><svg viewBox="0 0 ${Number(evidence.image_width_pixels)} ${Number(evidence.image_height_pixels)}" aria-label="Observed and projected calibration points">${circles(evidence.solve_observed_pixels, "observed solve")}${circles(evidence.solve_projected_pixels, "projected solve")}${circles(evidence.validation_observed_pixels, "observed validation")}${circles(evidence.validation_projected_pixels, "projected validation")}</svg></div><figcaption>Evidence overlay: filled circles are observed points; rings are projections. Cyan = solve, amber = validation.</figcaption></figure>`;
}

function cameraSummary(camera, expanded = true) {
  const i = camera.intrinsics;
  const p = camera.pose;
  if (!i || !p || !Array.isArray(p.matrix)) return `<div class="warning">No current reviewed pose is available for this camera.</div>`;
  const rows = p.matrix.map(row => `<tr>${row.map(value => `<td>${Number(value).toFixed(6)}</td>`).join("")}</tr>`).join("");
  return `<div class="camera-summary ${expanded ? "expanded" : ""}">
    <div><h3>Intrinsics</h3><dl class="details"><dt>Model</dt><dd>${escapeHtml(i.model)}</dd><dt>Resolution</dt><dd>${i.resolution[0]} × ${i.resolution[1]} px</dd><dt>Focal length</dt><dd>fx ${i.fx_pixels.toFixed(6)} · fy ${i.fy_pixels.toFixed(6)} px</dd><dt>Principal point</dt><dd>cx ${i.cx_pixels.toFixed(6)} · cy ${i.cy_pixels.toFixed(6)} px</dd><dt>Distortion</dt><dd>${i.distortion.map(value => Number(value).toFixed(9)).join(", ")}</dd></dl></div>
    <div><h3>Fixed-centre orientation in world frame</h3><dl class="details"><dt>Surveyed position (m)</dt><dd>x ${p.position_metres[0].toFixed(6)} · y ${p.position_metres[1].toFixed(6)} · z ${p.position_metres[2].toFixed(6)}</dd><dt>Solved orientation</dt><dd>yaw ${p.orientation_zyx_degrees.yaw.toFixed(4)}° · pitch ${p.orientation_zyx_degrees.pitch.toFixed(4)}° · roll ${p.orientation_zyx_degrees.roll.toFixed(4)}°</dd></dl><details ${expanded ? "" : ""}><summary>Exact T_world_from_camera matrix</summary><table class="matrix"><tbody>${rows}</tbody></table></details></div>
  </div>`;
}

function reconstructionPage() {
  if (workflowCapability("reconstruction")?.state === "not_provisioned") {
    return sceneNotProvisioned("Static reconstruction", "reconstruction", "calibration", "Return to Calibration & pose");
  }
  const workflow = asObject(state.status?.operator) || {};
  const geometry = asObject(workflow.geometry) || {};
  const inputIssues = asArray(workflow.input_issues);
  const calibrationWarnings = asArray(workflow.calibration_warnings);
  const cameras = asArray(workflow.cameras);
  const reconstructing = workflow.active_action?.action === "all-camera-static-reconstruction";
  return `<article class="card wide"><div class="section-heading"><div><h2>Camera inputs</h2><p>Static reconstruction uses these exact accepted intrinsics and world-frame poses.</p></div>${statusBadge(workflow.inputs_ready ? "complete" : "attention")}</div>
    ${inputIssues.length ? `<div class="warning"><strong>Reconstruction is blocked.</strong> ${inputIssues.map(escapeHtml).join(" ")} Return to Calibration & pose and rework the affected camera.</div>` : ""}
    ${calibrationWarnings.length ? `<div class="warning"><strong>Operator-qualified calibration.</strong> ${calibrationWarnings.map(escapeHtml).join(" ")} The automated failure remains recorded in the reconstruction input manifest.</div>` : ""}
    <div class="accordion-list">${cameras.map(camera => `<details><summary>${escapeHtml(camera.camera_id)} ${statusBadge(camera.ready ? "complete" : "attention")}</summary>${cameraSummary(camera)}</details>`).join("")}</div>
  </article>
  ${reconstructing ? `<article class="card wide progress-card"><strong>Static reconstruction is running.</strong><p>DA3 inference, point-cloud export and verification normally take about one to two minutes. Preview will become available only after all three complete.</p></article>` : ""}
  <article class="card action-card"><h2>Build combined point cloud</h2><p>Run pose-conditioned static reconstruction for every ready camera, then export and verify the combined geometry.</p><button id="run-reconstruction" ${geometry.can_run ? "" : "disabled"}>${reconstructing ? "Static reconstruction running…" : "Run static reconstruction"}</button></article>
  <article class="card action-card"><h2>Geometry review</h2><p>Open the current combined point cloud in Rerun. Approve it only after visual inspection.</p><div class="button-row"><button id="open-geometry" ${geometry.can_preview ? "" : "disabled"}>Open Rerun preview</button><button id="approve-geometry" class="secondary" ${geometry.previewed && !geometry.approved ? "" : "disabled"}>${geometry.approved ? "Geometry approved" : "Approve geometry"}</button></div></article>
  ${jobsCard()}`;
}

function floorPage() {
  if (workflowCapability("floor")?.state === "not_provisioned") {
    return sceneNotProvisioned("Floor refinement", "floor", "reconstruction", "Return to Static reconstruction");
  }
  const floor = state.status.operator.floor;
  const active = state.status.operator.active_action;
  const inputIssues = asArray(state.status.operator.input_issues);
  const floorRunning = active?.action === "floor-completion";
  const previewRunning = active?.action === "build-and-open-floor-preview";
  return `<article class="card wide"><div class="section-heading"><div><h2>Authoritative floor at Z=0</h2><p>Create the deterministic floor derivative only after the combined geometry is approved. Original coloured point-cloud samples remain unchanged.</p></div>${statusBadge(floor.approved ? "approved" : floor.can_generate ? "ready" : "blocked")}</div>
    ${!floor.can_generate && !floor.approved ? `<div class="warning">${inputIssues.length ? "Resolve the current scene inputs before continuing." : "Approve the combined geometry in Static reconstruction before generating the floor result."}</div>` : ""}
  </article>
  ${floorRunning || previewRunning ? `<article class="card wide progress-card"><strong>${floorRunning ? "Floor generation" : "Final preview"} is running.</strong><p>The current action is protected from duplicate clicks. Existing immutable results are not overwritten.</p></article>` : ""}
  <article class="card action-card"><h2>Generate final point cloud</h2><p>Add the configured authoritative floor plane and produce a new immutable result.</p><button id="run-floor" ${floor.can_generate ? "" : "disabled"}>${floorRunning ? "Generating floor-completed result…" : "Generate floor-completed result"}</button></article>
  <article class="card action-card"><h2>Final review</h2><p>Build and open the newest result, or inspect the current final recording. Approve only after checking it in Rerun.</p><div class="button-stack"><button id="build-floor-preview" ${floor.can_build_preview ? "" : "disabled"} data-floor-job="${escapeHtml(floor.current_floor_job_id || "")}">${previewRunning ? "Building final preview…" : "Build & open newest preview"}</button><button id="open-floor" class="secondary" ${floor.can_preview ? "" : "disabled"}>${floor.opening_latest_approved ? "Open latest approved final preview" : "Open current final preview"}</button><button id="approve-floor" class="secondary" ${floor.previewed && !floor.approved ? "" : "disabled"}>${floor.approved ? "Final result approved" : "Approve final result"}</button></div></article>
  ${jobsCard()}`;
}

function resultsPage() {
  const workflow = state.status.operator;
  if (workflowCapability("final_review")?.state === "not_provisioned") {
    return sceneNotProvisioned("Final review", "final_review", "floor", "Return to Floor refinement");
  }
  const liveAvailable = workflowCapability("live_operations")?.state === "available";
  const updatesAvailable = workflowCapability("scene_updates")?.state === "available";
  const approvedActions = `${liveAvailable ? `<a class="button-link" href="${scenePage("live")}">Open Live operations</a>` : ""}${updatesAvailable ? `<a class="button-link secondary" href="${scenePage("updates")}">Manage scene updates</a>` : ""}`;
  return `<article class="card hero wide"><p class="eyebrow">Workflow result</p><h2>${workflow.floor.approved ? "Final result approved" : "Final review is still required"}</h2><p>${workflow.floor.approved ? "The current floor-completed point cloud has been previewed and approved." : "Open the final Rerun preview on the Floor refinement page and approve it when it meets your needs."}</p><div class="button-row">${workflow.floor.approved ? approvedActions : `<a class="button-link" href="${scenePage("floor")}">Go to floor review</a>`}</div></article>
    ${overlapEditor()}
    ${cameraRoster()}`;
}

function liveOperationsPage() {
  if (workflowCapability("live_operations")?.state === "not_provisioned") {
    return sceneNotProvisioned("Live operations", "live_operations", "results", "Return to Final review");
  }
  const integrated = asObject(state.status?.live_operations) || {};
  const worker = asObject(integrated.worker) || {};
  const coordination = asObject(state.status?.scene_updates?.live_coordination) || {};
  const active = worker.active === true;
  const mode = worker.active_mode;
  const pending = asObject(worker.pending_run);
  const blockers = asArray(integrated.blockers);
  const saved = asArray(worker.saved_recordings);
  const recent = asArray(worker.recent_live_runs);
  const activePanel = active ? `<article class="card wide live-active"><div class="section-heading"><div><p class="eyebrow">${mode === "recording" ? "Replayable Recording" : "Live Service"}</p><h2>${mode === "recording" ? "Recording is running" : "Live tracking is running"}</h2><p>${mode === "recording" ? "Scene updates are deferred until this recording stops." : "A scheduled scene update will pause Live, adopt the verified result, and restart Live automatically."}</p></div>${statusBadge("running")}</div><div class="button-row"><button id="open-live-rerun" class="secondary">Open Rerun 3D</button><button id="stop-live-operations" class="danger">${mode === "recording" ? "Stop Recording" : "Stop Live Service"}</button></div></article>` : "";
  const startPanel = !active && !pending ? `<article class="card wide"><div class="section-heading"><div><p class="eyebrow">Post-review operations</p><h2>Choose one operating mode</h2><p>Both modes use this current run's approvals, camera policy, native Rerun facility view and scene binding. Retained historical outputs do not unlock Live automatically.</p></div>${statusBadge(integrated.eligible ? "ready" : "blocked")}</div>${blockers.length ? `<div class="warning"><strong>Live operations are locked.</strong><ul>${blockers.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}<div class="live-mode-grid"><button id="start-live-service" class="live-mode" ${integrated.eligible ? "" : "disabled"}><strong>Start Live Service</strong><span>Live Rerun plus compact 1 Hz count, track and XY history. No replay archive.</span></button><button id="start-live-recording" class="live-mode recording" ${integrated.eligible && worker.recording_available ? "" : "disabled"}><strong>Start Replayable Recording</strong><span>Full Rerun timeline, decision journals and camera footage. Save or delete after stopping.</span></button></div></article>` : "";
  const pendingPanel = pending ? `<article class="card wide warning-card"><p class="eyebrow">Recording disposition required</p><h2>${pending.state === "recovery_required" ? "Run recovery required" : "Recording is ready"}</h2><p>${escapeHtml(pending.session_id)} · ${escapeHtml(formatBytes(pending.byte_count))}</p><div class="button-row"><button class="secondary" id="view-pending-recording" ${pending.mode === "recording" && pending.state !== "recovery_required" ? "" : "disabled"}>View recording</button><input id="recording-label" maxlength="80" placeholder="Recording name" ${pending.state === "recovery_required" ? "disabled" : ""}><button id="save-pending-recording" ${pending.state === "recovery_required" ? "disabled" : ""}>Name &amp; save</button><button id="delete-pending-recording" class="danger">Delete exact run…</button></div></article>` : "";
  const coordinationPanel = coordination.state && coordination.state !== "idle" ? `<article class="card wide progress-card"><strong>${escapeHtml(humanState(coordination.state))}</strong><p>${escapeHtml(coordination.message || "Live/scene-update coordination is active.")}</p>${coordination.resume_requested ? '<button id="cancel-live-resume" class="secondary">Do not restart Live</button>' : ""}</article>` : "";
  const savedRows = saved.map(run => `<div class="history-row"><div><strong>${escapeHtml(run.label)}</strong><small>${escapeHtml(new Date(run.started_at_utc).toLocaleString())} · ${escapeHtml(formatBytes(run.byte_count))}</small></div><button class="secondary view-saved-recording" data-session="${escapeHtml(run.session_id)}">View</button></div>`).join("");
  const liveRows = recent.map(run => `<div class="history-row"><div><strong>Live Service</strong><small>${escapeHtml(new Date(run.started_at_utc).toLocaleString())} · ${escapeHtml(formatBytes(run.byte_count))} · ${escapeHtml(humanState(run.stop_reason || "stopped"))}</small></div></div>`).join("");
  return `${coordinationPanel}${activePanel}${startPanel}${pendingPanel}<article class="card wide"><h2>Saved recordings</h2><div class="history-list">${savedRows || '<p class="muted">No saved recordings yet.</p>'}</div></article><article class="card wide"><h2>Recent compact Live runs</h2><div class="history-list">${liveRows || '<p class="muted">No completed Live runs yet.</p>'}</div></article><details class="card wide"><summary>Diagnostics &amp; engineering controls</summary><div class="button-row"><button id="diagnostic-open-rerun" class="secondary" ${worker.output_directory ? "" : "disabled"}>Open Rerun 3D</button><button id="diagnostic-reset-trails" class="secondary" ${active ? "" : "disabled"}>Reset trails</button><button id="diagnostic-export" class="secondary" ${worker.output_directory ? "" : "disabled"}>Evidence snapshot</button></div></details>`;
}

const ACTIVITY_VISIBLE_LIMIT = 5;

function activityTime(value) {
  const parsed = Date.parse(value || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function activityRowAttributes(group, timestamp, searchText, archived) {
  if (!archived) return "";
  return ` data-activity-row="${escapeHtml(group)}" data-activity-time="${activityTime(timestamp)}" data-activity-search="${escapeHtml(searchText.toLowerCase())}"`;
}

function sceneUpdateActivityRow(event, archived = false) {
  const label = event.kind.replaceAll("-", " ");
  const search = `${label} ${event.message}`;
  return `<div class="job-row${archived ? " activity-row" : ""}"${activityRowAttributes("updates", event.at_utc, search, archived)}><div><strong>${escapeHtml(label)}</strong><small class="activity-message" title="${escapeHtml(event.message)}">${escapeHtml(new Date(event.at_utc).toLocaleString())} · ${escapeHtml(event.message)}</small></div></div>`;
}

function workflowActivityRow(job, labels, archived = false) {
  const label = labels[job.action] || "Workflow action";
  const timestamp = job.completed_at_utc || job.submitted_at_utc;
  const message = job.error_message || job.job_id;
  const search = `${label} ${job.state} ${message}`;
  return `<div class="job-row${archived ? " activity-row" : ""}"${activityRowAttributes("jobs", timestamp, search, archived)}><div><strong>${escapeHtml(label)}</strong><small class="activity-message" title="${escapeHtml(message)}">${timestamp ? `${escapeHtml(new Date(timestamp).toLocaleString())} · ` : ""}${escapeHtml(message)}</small></div>${statusBadge(job.state)}</div>`;
}

function activityArchive(group, older, rowBuilder) {
  if (!older.length) return "";
  return `<details class="older-history activity-archive"><summary>Search older activity (${older.length})</summary><div class="older-history-body"><div class="older-toolbar activity-toolbar"><label>Search<input type="search" placeholder="Status, action or message" data-activity-query="${group}"></label><label>From<input type="datetime-local" data-activity-from="${group}"></label><label>To<input type="datetime-local" data-activity-to="${group}"></label><button class="secondary clear-activity-filter" data-activity-group="${group}">Clear</button></div><p class="muted activity-match-count" data-activity-count="${group}">${older.length} older entries</p><div class="job-list">${older.map(item => rowBuilder(item, true)).join("")}</div><p class="muted activity-filter-empty hidden" data-activity-empty="${group}">No older activity matches these filters.</p></div></details>`;
}

function activityPanel(title, group, items, rowBuilder, emptyMessage) {
  if (!items.length) return `<article class="card wide"><h2>${title}</h2><p class="muted">${emptyMessage}</p></article>`;
  const ordered = items.slice().sort((left, right) => activityTime(right.at_utc || right.completed_at_utc || right.submitted_at_utc) - activityTime(left.at_utc || left.completed_at_utc || left.submitted_at_utc));
  const recent = ordered.slice(0, ACTIVITY_VISIBLE_LIMIT);
  const older = ordered.slice(ACTIVITY_VISIBLE_LIMIT);
  return `<article class="card wide"><h2>${title}</h2><div class="job-list">${recent.map(item => rowBuilder(item, false)).join("")}</div>${activityArchive(group, older, rowBuilder)}</article>`;
}

function sceneUpdatesPage() {
  if (workflowCapability("scene_updates")?.state === "not_provisioned") {
    return sceneNotProvisioned("Scene updates", "scene_updates", "results", "Return to Final review");
  }
  const updates = state.status.scene_updates;
  const cameraCount = asArray(state.status?.camera_roster).filter(camera => camera.enabled).length;
  const cameraLabel = `${cameraCount} configured ${cameraCount === 1 ? "camera" : "cameras"}`;
  if (!updates.unlocked_at_utc) return `<article class="card hero wide"><p class="eyebrow">Protected capability</p><h2>Scene updates are locked</h2><p>Complete the first manual final review and approve the floor-completed result before recurring reconstruction can be enabled.</p><a class="button-link" href="${scenePage("results")}">Return to final review</a></article>`;
  const schedule = updates.schedule;
  const weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const weekdayInputs = weekdays.map((label, index) => `<label class="check-option"><input type="checkbox" name="update-weekday" value="${index}" ${schedule.weekdays.includes(index) ? "checked" : ""}><span>${label.slice(0, 3)}</span></label>`).join("");
  const candidate = updates.manual_candidate;
  const liveWorker = asObject(state.status?.live_operations?.worker) || {};
  const liveActive = liveWorker.active === true;
  const rollback = (updates.rollback_results || []).map(result => `<article class="past-version"><div class="past-version-main"><div class="past-version-title"><div><span class="version-kind">Accepted result</span><strong>${escapeHtml(new Date(result.adopted_at_utc).toLocaleString())}</strong></div>${result.initial_manual_result ? '<span class="state approved">Initial manual</span>' : ""}</div><small>${escapeHtml(result.trigger_mode.replaceAll("_", " "))}</small></div><div class="past-version-actions"><button class="secondary rollback-scene-update" data-result="${escapeHtml(result.result_id)}">Restore this result</button></div></article>`).join("");
  return `<article class="card hero wide"><p class="eyebrow">Post-review capability</p><h2>Keep the static scene current</h2><p>Every update captures the ${escapeHtml(cameraLabel)}, rebuilds DA3 geometry, regenerates the floor result and creates a new final Rerun recording. Intrinsics and camera poses remain unchanged.</p><div class="metric-grid"><div class="metric"><strong>${schedule.enabled ? "On" : "Off"}</strong><span>Automatic updates</span></div><div class="metric"><strong>${escapeHtml(schedule.enabled ? schedule.mode : "Manual")}</strong><span>Current mode</span></div><div class="metric"><strong>${updates.next_due_at_utc ? escapeHtml(new Date(updates.next_due_at_utc).toLocaleString()) : "Not scheduled"}</strong><span>Next occurrence</span></div></div>${!updates.available ? '<div class="warning"><strong>Live updates are unavailable in this launch.</strong> Restart the combined console with live RTSP capture enabled.</div>' : ""}${liveActive ? `<div class="warning"><strong>${liveWorker.active_mode === "recording" ? "Recording has priority." : "Live coordination is active."}</strong> ${liveWorker.active_mode === "recording" ? "Scheduled updates are deferred until Recording stops; manual updates are blocked." : "Scheduled updates pause and resume Live automatically; stop Live before starting a manual update."}</div>` : ""}</article>
  <article class="card action-card"><h2>1. Manual update</h2><p>Capture one fresh frame per camera and build a candidate. The current accepted result stays active until you open and approve the candidate.</p><button id="run-scene-update" ${updates.available && !updates.pipeline_busy && !liveActive ? "" : "disabled"}>Build manual candidate</button></article>
  <article class="card action-card"><h2>2. Automatic schedule</h2><p>Daily and interval runs replace the current result only after the complete processing chain succeeds. Turn this off at any time.</p><label class="field"><span>Mode</span><select id="update-mode"><option value="daily" ${schedule.mode === "daily" ? "selected" : ""}>Set local time</option><option value="interval" ${schedule.mode === "interval" ? "selected" : ""}>Regular interval</option></select></label><label class="check-row"><input id="update-enabled" type="checkbox" ${schedule.enabled ? "checked" : ""}><span>Enable automatic updates while this console is running</span></label><div id="daily-settings"><label class="field"><span>Local time</span><input id="update-daily-time" type="time" value="${escapeHtml(schedule.daily_time)}"></label><div class="field"><span>Days</span><div class="check-options">${weekdayInputs}</div></div></div><div id="interval-settings"><label class="field"><span>Repeat every</span><select id="update-interval"><option value="1800">30 minutes</option><option value="3600">1 hour</option><option value="10800">3 hours</option><option value="21600">6 hours</option><option value="43200">12 hours</option><option value="86400">1 day</option><option value="604800">1 week</option></select></label><label class="field"><span>Median frames per camera</span><select id="update-frame-count">${[3,5,7,9,11].map(value => `<option value="${value}" ${schedule.median_frame_count === value ? "selected" : ""}>${value} frames</option>`).join("")}</select></label><label class="field"><span>Seconds between frame sets</span><input id="update-spacing" type="number" min="10" max="600" step="10" value="${schedule.median_spacing_seconds}"></label><p class="muted">All configured cameras are captured together at each step. A per-camera median image reduces people and other temporary foreground objects before DA3.</p></div><button id="save-update-schedule">Save schedule</button></article>
  ${candidate ? `<article class="card wide"><div class="section-heading"><div><p class="eyebrow">Manual review required</p><h2>New candidate is ready</h2><p>${escapeHtml(candidate.result_id)} · ${escapeHtml(new Date(candidate.adopted_at_utc).toLocaleString())}</p></div>${statusBadge(updates.manual_candidate_previewed ? "ready_for_review" : "pending")}</div><div class="button-row"><button id="open-update-candidate" data-result="${escapeHtml(candidate.result_id)}">Open Rerun candidate</button><button id="adopt-update-candidate" class="secondary" data-result="${escapeHtml(candidate.result_id)}" ${updates.manual_candidate_previewed ? "" : "disabled"}>Approve and use candidate</button></div></article>` : ""}
  <article class="card wide"><div class="section-heading"><div><h2>Rollback</h2><p>The initial manual result and up to three recently superseded accepted results remain directly restorable.</p></div></div><div class="history-list">${rollback || '<p class="muted">No superseded accepted results yet.</p>'}</div></article>
  ${activityPanel("Recent update activity", "updates", updates.events || [], sceneUpdateActivityRow, "No scene-update activity yet.")}${jobsCard()}`;
}

function jobsCard() {
  const jobs = state.status.jobs;
  const labels = { "all-camera-static-reconstruction": "Static reconstruction", "floor-completion": "Floor generation", "build-and-open-floor-preview": "Final preview", "full-scene-update": "Complete scene update" };
  return activityPanel("Recent activity", "jobs", jobs, (job, archived) => workflowActivityRow(job, labels, archived), "No processing actions have run in this session.");
}

function filterActivity(group) {
  const query = (document.querySelector(`[data-activity-query="${group}"]`)?.value || "").trim().toLowerCase();
  const fromValue = document.querySelector(`[data-activity-from="${group}"]`)?.value || "";
  const toValue = document.querySelector(`[data-activity-to="${group}"]`)?.value || "";
  const from = fromValue ? new Date(fromValue).getTime() : Number.NEGATIVE_INFINITY;
  const to = toValue ? new Date(toValue).getTime() + 59999 : Number.POSITIVE_INFINITY;
  const rows = Array.from(document.querySelectorAll(`[data-activity-row="${group}"]`));
  let visible = 0;
  rows.forEach(row => {
    const time = Number(row.dataset.activityTime);
    const matches = row.dataset.activitySearch.includes(query) && time >= from && time <= to;
    row.classList.toggle("activity-hidden", !matches);
    if (matches) visible += 1;
  });
  const count = document.querySelector(`[data-activity-count="${group}"]`);
  if (count) count.textContent = `${visible} of ${rows.length} older entries`;
  document.querySelector(`[data-activity-empty="${group}"]`)?.classList.toggle("hidden", visible !== 0);
}

function clearActivityFilter(group) {
  document.querySelectorAll(`[data-activity-query="${group}"], [data-activity-from="${group}"], [data-activity-to="${group}"]`).forEach(input => { input.value = ""; });
  filterActivity(group);
}

function renderContent() {
  const page = state.pages.find(item => item.page_id === selectedPageId()) || state.pages[0];
  let html = "";
  if (page.page_id === "setup") html = projectOverview();
  if (page.page_id === "artifacts") html = artifactVersionsPage();
  if (page.page_id === "facility") html = `<article class="card wide"><h2>Facility scale check</h2><p>The selected facility result remains the current scene input. Any new registration revision must add one horizontal and one vertical physical scale reference; metric XY and export use the mean of their two pixels-per-metre values.</p></article>${lensGroupEditor()}${embeddedTool(page, "Facility registration is not configured in this launch.")}`;
  if (page.page_id === "capture") html = embeddedTool(page, "Live capture is not enabled in this launch.");
  if (page.page_id === "calibration") html = calibrationPage(page);
  if (page.page_id === "reconstruction") html = reconstructionPage();
  if (page.page_id === "floor") html = floorPage();
  if (page.page_id === "results") html = resultsPage();
  if (page.page_id === "live") html = liveOperationsPage();
  if (page.page_id === "updates") html = sceneUpdatesPage();
  const shared = asObject(state.status?.shared_processing) || {};
  const active = asObject(shared.active);
  const queued = asArray(shared.queue);
  const waitingHere = queued.filter(item => item.scene_uuid === selectedSceneId()).length;
  const sharedBanner = active && active.scene_uuid !== selectedSceneId()
    ? `<article class="card wide progress-card"><strong>Another scene is using heavy processing.</strong><p>You can keep editing this scene. New processing will wait safely until this computer is free.</p></article>`
    : waitingHere ? `<article class="card wide progress-card"><strong>${waitingHere} processing ${waitingHere === 1 ? "action is" : "actions are"} waiting.</strong><p>This scene will continue automatically when the current heavy operation finishes.</p></article>` : "";
  document.getElementById("content").innerHTML = `${sharedBanner}${html}`;
  if (page.page_id === "updates") {
    const schedule = state.status.scene_updates.schedule;
    const interval = document.getElementById("update-interval");
    if (interval) interval.value = String(schedule.interval_seconds);
    updateScheduleFields();
  }
  bindActions();
}

function bindActions() {
  document.querySelectorAll(".camera-tab").forEach(button => button.addEventListener("click", () => { state.selectedCamera = button.dataset.camera; renderContent(); }));
  document.getElementById("determine-intrinsics")?.addEventListener("click", determineIntrinsics);
  document.getElementById("calibrate-camera-now")?.addEventListener("click", calibrateSelectedCamera);
  document.getElementById("review-calibration")?.addEventListener("click", event => reviewSelectedCalibration(event.currentTarget.dataset.attempt));
  document.getElementById("override-calibration")?.addEventListener("click", event => overrideSelectedCalibration(event.currentTarget.dataset.attempt));
  document.getElementById("revise-linked-points")?.addEventListener("click", () => document.getElementById("correspondence-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  document.querySelectorAll(".history-filter").forEach(button => button.addEventListener("click", () => { state.artifactSection = button.dataset.section; renderContent(); }));
  document.querySelectorAll("[data-activity-query], [data-activity-from], [data-activity-to]").forEach(input => input.addEventListener("input", () => filterActivity(input.dataset.activityQuery || input.dataset.activityFrom || input.dataset.activityTo)));
  document.querySelectorAll(".clear-activity-filter").forEach(button => button.addEventListener("click", () => clearActivityFilter(button.dataset.activityGroup)));
  document.getElementById("run-reconstruction")?.addEventListener("click", () => startJob("/api/jobs/reconstruction", { job_id: uniqueId("reconstruction") }, "Static reconstruction"));
  document.getElementById("run-floor")?.addEventListener("click", () => startJob("/api/jobs/floor", { job_id: uniqueId("floor") }, "Floor generation"));
  document.getElementById("build-floor-preview")?.addEventListener("click", event => startJob("/api/jobs/floor-preview", { job_id: uniqueId("floor-preview"), floor_job_id: event.currentTarget.dataset.floorJob }, "Final preview"));
  document.getElementById("open-geometry")?.addEventListener("click", () => openPreview("geometry"));
  document.getElementById("open-floor")?.addEventListener("click", () => openPreview("floor"));
  document.getElementById("approve-geometry")?.addEventListener("click", () => approve("geometry"));
  document.getElementById("approve-floor")?.addEventListener("click", () => approve("floor"));
  document.getElementById("fresh-session")?.addEventListener("click", startFreshSession);
  document.getElementById("run-scene-update")?.addEventListener("click", startManualSceneUpdate);
  document.getElementById("save-update-schedule")?.addEventListener("click", saveSceneUpdateSchedule);
  document.getElementById("update-mode")?.addEventListener("change", updateScheduleFields);
  document.getElementById("open-update-candidate")?.addEventListener("click", event => openSceneUpdateCandidate(event.currentTarget.dataset.result));
  document.getElementById("adopt-update-candidate")?.addEventListener("click", event => adoptSceneUpdateCandidate(event.currentTarget.dataset.result));
  document.querySelectorAll(".rollback-scene-update").forEach(button => button.addEventListener("click", () => rollbackSceneUpdate(button.dataset.result)));
  document.getElementById("start-live-service")?.addEventListener("click", () => startLiveOperations("live"));
  document.getElementById("start-live-recording")?.addEventListener("click", () => startLiveOperations("recording"));
  document.getElementById("stop-live-operations")?.addEventListener("click", stopLiveOperations);
  document.getElementById("open-live-rerun")?.addEventListener("click", () => liveAction("open-rerun"));
  document.getElementById("view-pending-recording")?.addEventListener("click", () => liveAction("view-recording", { session_id: state.status.live_operations.worker.pending_run.session_id }));
  document.getElementById("save-pending-recording")?.addEventListener("click", () => liveAction("save-recording", { session_id: state.status.live_operations.worker.pending_run.session_id, label: document.getElementById("recording-label").value }));
  document.getElementById("delete-pending-recording")?.addEventListener("click", deletePendingRecording);
  document.querySelectorAll(".view-saved-recording").forEach(button => button.addEventListener("click", () => liveAction("view-recording", { session_id: button.dataset.session })));
  document.getElementById("cancel-live-resume")?.addEventListener("click", cancelLiveResume);
  document.getElementById("diagnostic-open-rerun")?.addEventListener("click", () => liveAction("open-rerun"));
  document.getElementById("diagnostic-reset-trails")?.addEventListener("click", () => liveAction("reset-trails"));
  document.getElementById("diagnostic-export")?.addEventListener("click", () => liveAction("export"));
  document.getElementById("save-lens-policy")?.addEventListener("click", () => saveCameraPolicy("lens"));
  document.getElementById("add-overlap-pair")?.addEventListener("click", addOverlapPair);
  document.querySelectorAll(".remove-overlap-pair").forEach(bindOverlapRemoval);
  document.getElementById("save-overlap-policy")?.addEventListener("click", () => saveCameraPolicy("overlap"));
  document.querySelectorAll(".rollback-camera-policy").forEach(button => button.addEventListener("click", () => rollbackCameraPolicy(Number(button.dataset.revision))));
  document.querySelectorAll(".verify-version").forEach(button => button.addEventListener("click", () => verifyVersion(button.dataset.artifact)));
  document.querySelectorAll(".select-version").forEach(button => button.addEventListener("click", () => selectVersion(button.dataset.artifact)));
  document.querySelectorAll(".archive-version").forEach(button => button.addEventListener("click", () => archiveVersion(button.dataset.artifact, button.dataset.archived === "true")));
  document.querySelectorAll(".delete-version").forEach(button => button.addEventListener("click", () => openDeleteDialog(button.dataset.artifact)));
  document.querySelectorAll(".batch-version-checkbox").forEach(input => input.addEventListener("change", () => updateBatchSelection(input.dataset.section)));
  document.querySelectorAll(".apply-history-filter").forEach(button => button.addEventListener("click", () => applyHistoryFilter(button.dataset.section)));
  document.querySelectorAll(".clear-history-filter").forEach(button => button.addEventListener("click", () => clearHistoryFilter(button.dataset.section)));
  document.querySelectorAll(".select-visible-history").forEach(button => button.addEventListener("click", () => selectVisibleHistory(button.dataset.section)));
  document.querySelectorAll(".batch-delete-history").forEach(button => button.addEventListener("click", () => openBatchDeleteDialog(button.dataset.section)));
}

async function startLiveOperations(mode) {
  try {
    await api("/api/live-operations/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode }) });
    notify(mode === "live" ? "Live Service is starting." : "Replayable Recording is starting.");
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function stopLiveOperations() {
  try {
    await api("/api/live-operations/stop", { method: "POST" });
    notify("Live operations stopped safely.");
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function liveAction(action, values = {}) {
  try {
    await api("/api/live-operations/action", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, ...values }) });
    notify(humanState(action));
    await Promise.all([refreshStatus(true), refreshArtifacts(false)]);
  } catch (error) { notify(error.message, true); }
}

async function deletePendingRecording() {
  const pending = state.status.live_operations.worker.pending_run;
  if (!window.confirm(`Permanently delete this exact XR02 run?\n\n${pending.session_id}\n\nThis cannot be recovered.`)) return;
  await liveAction("delete-recording", { session_id: pending.session_id, confirmation: `DELETE ${pending.session_id}` });
}

async function cancelLiveResume() {
  try {
    await api("/api/live-operations/cancel-resume", { method: "POST" });
    notify("Automatic Live restart cancelled.");
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function determineIntrinsics() {
  try {
    await api("/api/calibration/determine-intrinsics", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("determine-intrinsics") }) });
    notify("Intrinsic profiles were determined and assigned for the active lens policy.");
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function calibrateSelectedCamera() {
  const cameraId = state.selectedCamera;
  try {
    await api(`/api/calibration/cameras/${encodeURIComponent(cameraId)}/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId(`calibrate-${cameraId}`) }) });
    notify(`${cameraId} calibration finished. Review the numeric result and evidence overlay.`);
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function reviewSelectedCalibration(attemptSha256) {
  const cameraId = state.selectedCamera;
  try {
    await api(`/api/calibration/cameras/${encodeURIComponent(cameraId)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId(`review-${cameraId}`), attempt_sha256: attemptSha256 }) });
    notify(`${cameraId} is ready for Static reconstruction after physical evidence review.`);
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function overrideSelectedCalibration(attemptSha256) {
  const cameraId = state.selectedCamera;
  const reason = document.getElementById("override-reason")?.value.trim() || "";
  const acknowledged = Boolean(document.getElementById("override-acknowledged")?.checked);
  try {
    await api(`/api/calibration/cameras/${encodeURIComponent(cameraId)}/override`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId(`override-${cameraId}`), attempt_sha256: attemptSha256, reason, acknowledged }) });
    notify(`${cameraId} is operator-accepted with a persistent reconstruction warning.`);
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function refreshArtifacts(render = false) {
  state.artifactCatalog = await api("/api/artifacts");
  if (render && selectedPageId() === "artifacts") renderContent();
}

function cameraPolicyPayload(kind) {
  const policy = JSON.parse(JSON.stringify(currentCameraPolicy()));
  if (kind === "lens") {
    const groups = new Map();
    document.querySelectorAll(".lens-policy-row").forEach(row => {
      const groupId = row.querySelector(".lens-group-id").value.trim();
      const lensModel = row.querySelector(".lens-model").value.trim();
      if (!groupId && !lensModel) return;
      if (!groupId || !lensModel) throw new Error("Every assigned camera needs both a lens group and lens model.");
      const existing = groups.get(groupId);
      if (existing && existing.lens_model !== lensModel) throw new Error(`Lens group ${groupId} has conflicting lens-model names.`);
      const group = existing || { group_id: groupId, lens_model: lensModel, camera_ids: [] };
      group.camera_ids.push(row.dataset.cameraId);
      groups.set(groupId, group);
    });
    policy.intrinsic_groups = [...groups.values()];
  } else {
    const overlapKeys = new Set([...document.querySelectorAll(".overlap-policy-row")].map(row => `${row.dataset.cameraA}::${row.dataset.cameraB}`));
    const cameraIds = asArray(policy.camera_ids);
    policy.overlap_pair_reviews = [];
    for (let left = 0; left < cameraIds.length; left += 1) {
      for (let right = left + 1; right < cameraIds.length; right += 1) {
        const cameraIdA = cameraIds[left];
        const cameraIdB = cameraIds[right];
        policy.overlap_pair_reviews.push({
          camera_id_a: cameraIdA,
          camera_id_b: cameraIdB,
          verdict: overlapKeys.has(`${cameraIdA}::${cameraIdB}`) ? "overlap" : "no_overlap"
        });
      }
    }
  }
  return policy;
}

async function saveCameraPolicy(kind) {
  try {
    const payload = cameraPolicyPayload(kind);
    const impact = await api("/api/camera-policy/impact", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!impact.changed) { notify("Camera policy already matches these values."); return; }
    let confirmed = true;
    if (impact.requires_confirmation) {
      const messages = [];
      if (impact.intrinsic_reprocessing_required) messages.push("Lens membership changed; future intrinsic processing must use the new revision.");
      if (impact.xr02_new_epoch_required) messages.push("Overlap changed; a new XR02 scene epoch is required.");
      messages.push("DA3 scene-cohort membership is unchanged.");
      confirmed = window.confirm(`Save a new camera-policy revision?\n\n${messages.join("\n")}`);
    }
    if (!confirmed) return;
    await api("/api/camera-policy/apply", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...payload, action_id: uniqueId(`camera-policy-${kind}`), expected_revision: state.status.camera_policy.active_revision, confirm_impacts: confirmed }) });
    notify(kind === "lens" ? "Lens groups saved as a new scene-policy revision." : "Pairwise overlap review saved as a new scene-policy revision.");
    await Promise.all([refreshStatus(false), refreshArtifacts(false)]);
    renderContent();
  } catch (error) { notify(error.message, true); }
}

async function rollbackCameraPolicy(targetRevision) {
  if (!window.confirm(`Make camera-policy revision ${targetRevision} current? The newer revision remains in history and downstream impacts will be recorded.`)) return;
  try {
    await api("/api/camera-policy/rollback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("rollback-camera-policy"), target_revision: targetRevision, expected_revision: state.status.camera_policy.active_revision, confirm_impacts: true }) });
    notify(`Camera-policy revision ${targetRevision} is current. No scene or artifact file was changed.`);
    await Promise.all([refreshStatus(false), refreshArtifacts(false)]);
    renderContent();
  } catch (error) { notify(error.message, true); }
}

async function verifyVersion(artifactId) {
  try {
    const result = await api("/api/artifacts/verify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("verify-version"), artifact_id: artifactId }) });
    notify(result.lifecycle === "available" || result.lifecycle === "archived" ? "File identity verified." : `This version is ${result.lifecycle}.`, !["available", "archived"].includes(result.lifecycle));
    await refreshArtifacts(true);
  } catch (error) { notify(error.message, true); }
}

async function selectVersion(artifactId) {
  try {
    const impact = await api(`/api/artifacts/${encodeURIComponent(artifactId)}/impact`);
    let confirmed = true;
    if (impact.requires_confirmation) confirmed = window.confirm(`${impact.message}\n\nLater workflow results to reset:\n${(impact.downstream_titles || impact.downstream_selections).join("\n")}`);
    if (!confirmed) return;
    await api("/api/artifacts/select", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("select-version"), artifact_id: artifactId, confirm_impacts: confirmed }) });
    notify("Current version updated. Affected workflow steps and approvals were reset safely.");
    await refreshStatus(false); await refreshArtifacts(true);
  } catch (error) { notify(error.message, true); }
}

async function archiveVersion(artifactId, archived) {
  const verb = archived ? "archive" : "restore";
  if (archived && !window.confirm("Archive this version from normal use? Its physical file will remain preserved.")) return;
  try {
    await api("/api/artifacts/archive", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId(verb), artifact_id: artifactId, archived }) });
    notify(archived ? "Version archived. Its file was preserved." : "Version restored and available for selection.");
    await refreshArtifacts(true);
  } catch (error) { notify(error.message, true); }
}

function historyRows(section) {
  return [...document.querySelectorAll(`#older-list-${section} .batch-version`)];
}

function applyHistoryFilter(section) {
  const fromValue = document.getElementById(`history-from-${section}`).value;
  const toValue = document.getElementById(`history-to-${section}`).value;
  const from = fromValue ? new Date(fromValue).getTime() : Number.NEGATIVE_INFINITY;
  const to = toValue ? new Date(toValue).getTime() : Number.POSITIVE_INFINITY;
  historyRows(section).forEach(row => {
    const visible = new Date(row.dataset.created).getTime() >= from && new Date(row.dataset.created).getTime() <= to;
    row.classList.toggle("timestamp-hidden", !visible);
    if (!visible) row.querySelector(".batch-version-checkbox").checked = false;
  });
  updateBatchSelection(section);
}

function clearHistoryFilter(section) {
  document.getElementById(`history-from-${section}`).value = "";
  document.getElementById(`history-to-${section}`).value = "";
  applyHistoryFilter(section);
}

function selectVisibleHistory(section) {
  const visibleInputs = historyRows(section).filter(row => !row.classList.contains("timestamp-hidden")).map(row => row.querySelector(".batch-version-checkbox"));
  const shouldSelect = visibleInputs.some(input => !input.checked);
  visibleInputs.forEach(input => { input.checked = shouldSelect; });
  updateBatchSelection(section);
}

function selectedHistoryIds(section) {
  return [...document.querySelectorAll(`.batch-version-checkbox[data-section="${section}"]:checked`)].map(input => input.dataset.artifact);
}

function updateBatchSelection(section) {
  const selected = selectedHistoryIds(section);
  const button = document.querySelector(`.batch-delete-history[data-section="${section}"]`);
  if (!button) return;
  button.disabled = selected.length === 0;
  button.textContent = selected.length ? `Delete ${selected.length} selected ${selected.length === 1 ? "file" : "files"}…` : "Delete selected files…";
}

async function openDeleteDialog(artifactId) {
  const dialog = document.getElementById("delete-dialog");
  const content = document.getElementById("delete-dialog-content");
  state.deleteImpact = null;
  content.innerHTML = `<h2>Checking deletion impact…</h2><p>This does not change any file.</p>`;
  dialog.showModal();
  try {
    const impact = await api(`/api/artifacts/${encodeURIComponent(artifactId)}/delete-impact`);
    state.deleteImpact = impact;
    const dependents = impact.dependents.length ? `<div class="dependency-list"><strong>Retained results that use this file</strong><ul>${impact.dependents.map(item => `<li>${escapeHtml(item.milestone_title)}${item.selected ? " (current)" : ""}</li>`).join("")}</ul></div>` : "";
    const blockers = impact.blockers.length ? `<div class="delete-blockers"><strong>This file cannot be deleted yet.</strong><ul>${impact.blockers.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : "";
    const confirmation = impact.allowed ? `<button id="confirm-delete" class="danger">Delete this file permanently</button>` : "";
    content.innerHTML = `<p class="eyebrow">Permanent deletion</p><h2>${escapeHtml(impact.display_name)}</h2><div class="delete-file-summary"><div><span>Exact file</span><strong>${escapeHtml(impact.path)}</strong></div><div><span>Size</span><strong>${escapeHtml(formatBytes(impact.byte_count))}</strong></div></div><div class="danger-note"><strong>Before you delete:</strong> ${escapeHtml(impact.warning)}</div>${dependents}${blockers}${confirmation}`;
    if (impact.allowed) {
      const button = document.getElementById("confirm-delete");
      button.addEventListener("click", () => deleteVersionPermanently(artifactId, impact.deletion_token));
    }
  } catch (error) {
    content.innerHTML = `<h2>Deletion check failed</h2><div class="delete-blockers">${escapeHtml(error.message)}</div>`;
  }
}

async function deleteVersionPermanently(artifactId, deletionToken) {
  const button = document.getElementById("confirm-delete");
  button.disabled = true;
  button.textContent = "Deleting exact file…";
  try {
    const result = await api("/api/artifacts/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("delete-version"), artifact_id: artifactId, deletion_token: deletionToken }) });
    document.getElementById("delete-dialog").close();
    notify(`Past file permanently deleted. ${formatBytes(result.deleted_byte_count)} removed.`);
    await refreshArtifacts(true);
  } catch (error) {
    notify(error.message, true);
    button.disabled = false;
    button.textContent = "Delete this file permanently";
  }
}

async function openBatchDeleteDialog(section) {
  const artifactIds = selectedHistoryIds(section);
  if (!artifactIds.length) return;
  const dialog = document.getElementById("delete-dialog");
  const content = document.getElementById("delete-dialog-content");
  content.innerHTML = `<h2>Checking ${artifactIds.length} selected files…</h2><p>This does not change any file.</p>`;
  dialog.showModal();
  try {
    const impact = await api("/api/artifacts/delete-impact", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ artifact_ids: artifactIds }) });
    state.deleteImpact = impact;
    const fileRows = impact.items.map(item => `<li><strong>${escapeHtml(item.display_name)}</strong><span>${escapeHtml(new Date((state.artifactCatalog.milestones.flatMap(milestone => milestone.versions).find(version => version.artifact_id === item.artifact_id) || {}).created_at || "").toLocaleString())} · ${escapeHtml(formatBytes(item.byte_count))}</span>${item.blockers.length ? `<small>${escapeHtml(item.blockers.join(" "))}</small>` : ""}</li>`).join("");
    const blockers = impact.all_allowed ? "" : `<div class="delete-blockers"><strong>The batch cannot be deleted yet.</strong><p>Remove blocked files from the selection or resolve the listed dependency first.</p></div>`;
    const button = impact.all_allowed ? `<button id="confirm-batch-delete" class="danger">Delete ${impact.artifact_count} files permanently</button>` : "";
    content.innerHTML = `<p class="eyebrow">Batch permanent deletion</p><h2>Review ${impact.artifact_count} selected files</h2><div class="delete-file-summary"><div><span>Total storage to remove</span><strong>${escapeHtml(formatBytes(impact.total_byte_count))}</strong></div></div><div class="danger-note"><strong>Before you delete:</strong> ${escapeHtml(impact.warning)}</div><ul class="batch-delete-list">${fileRows}</ul>${blockers}${button}`;
    document.getElementById("confirm-batch-delete")?.addEventListener("click", () => deleteVersionBatch(impact.items));
  } catch (error) {
    content.innerHTML = `<h2>Batch deletion check failed</h2><div class="delete-blockers">${escapeHtml(error.message)}</div>`;
  }
}

async function deleteVersionBatch(items) {
  const button = document.getElementById("confirm-batch-delete");
  button.disabled = true;
  button.textContent = "Deleting selected files…";
  try {
    const deletionItems = items.map(item => ({ artifact_id: item.artifact_id, deletion_token: item.deletion_token }));
    const result = await api("/api/artifacts/delete-batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("delete-batch"), items: deletionItems }) });
    document.getElementById("delete-dialog").close();
    notify(`${result.deleted_artifact_count} past files permanently deleted. ${formatBytes(result.deleted_byte_count)} removed.`);
    await refreshArtifacts(true);
  } catch (error) {
    notify(error.message, true);
    button.disabled = false;
    button.textContent = "Delete selected files permanently";
  }
}

async function startJob(url, payload, label) {
  try {
    await api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    notify(`${label} started.`); await refreshStatus(true); monitorJob(payload.job_id, label);
  } catch (error) { notify(error.message, true); }
}

async function monitorJob(jobId, label) {
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (["complete", "failed", "cancelled"].includes(job.state)) {
      notify(`${label}: ${humanState(job.state)}.`, job.state === "failed");
      await refreshStatus(true); return;
    }
    await refreshStatus(true);
    window.setTimeout(() => monitorJob(jobId, label), 1500);
  } catch (error) { notify(error.message, true); }
}

async function openPreview(target) {
  const workflow = state.status.operator[target];
  const artifactId = target === "floor" ? workflow.open_artifact_id : workflow.artifact_id;
  try {
    await api("/api/rerun/launch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId(`open-${target}`), artifact_id: artifactId }) });
    notify(`${target === "geometry" ? "Geometry" : "Final"} preview opened in Rerun.`); await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function startFreshSession() {
  if (!window.confirm("Archive the current workflow activity and start a fresh demonstration run? Generated artifacts will be preserved.")) return;
  try {
    await api("/api/session/fresh", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("archive-session"), session_id: uniqueId("demo") }) });
    notify("A fresh demonstration run is ready. Previous results were preserved in the archive.");
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function approve(target) {
  try {
    await api("/api/approve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId(`approve-${target}`), target }) });
    notify(target === "geometry" ? "Geometry approved. Floor refinement is now available." : "Final result approved."); await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

function updateScheduleFields() {
  const mode = document.getElementById("update-mode")?.value;
  document.getElementById("daily-settings")?.classList.toggle("hidden", mode !== "daily");
  document.getElementById("interval-settings")?.classList.toggle("hidden", mode !== "interval");
}

async function saveSceneUpdateSchedule() {
  const mode = document.getElementById("update-mode").value;
  const weekdays = [...document.querySelectorAll('input[name="update-weekday"]:checked')].map(input => Number(input.value));
  const payload = {
    enabled: document.getElementById("update-enabled").checked,
    mode,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Singapore",
    daily_time: document.getElementById("update-daily-time").value,
    weekdays,
    interval_seconds: Number(document.getElementById("update-interval").value),
    median_frame_count: Number(document.getElementById("update-frame-count").value),
    median_spacing_seconds: Number(document.getElementById("update-spacing").value)
  };
  try {
    await api("/api/scene-updates/configure", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    notify(payload.enabled ? "Automatic scene updates enabled." : "Automatic scene updates switched off.");
    await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function startManualSceneUpdate() {
  const updateId = uniqueId("manual-update");
  try {
    await api("/api/scene-updates/manual", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ update_id: updateId }) });
    notify("Manual scene update started. The current accepted result will remain active.");
    await refreshStatus(true); monitorJob(updateId, "Manual scene update");
  } catch (error) { notify(error.message, true); }
}

async function openSceneUpdateCandidate(resultId) {
  try {
    await api("/api/scene-updates/candidate/open", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("open-update"), result_id: resultId }) });
    notify("Manual candidate opened in Rerun. Approve it only after review."); await refreshStatus(true);
  } catch (error) { notify(error.message, true); }
}

async function adoptSceneUpdateCandidate(resultId) {
  try {
    await api("/api/scene-updates/candidate/adopt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("adopt-update"), result_id: resultId }) });
    notify("Manual scene update approved and adopted."); await Promise.all([refreshStatus(true), refreshArtifacts(false)]);
  } catch (error) { notify(error.message, true); }
}

async function rollbackSceneUpdate(resultId) {
  if (!window.confirm("Restore this previously accepted result as the current scene? The latest result remains retained in history.")) return;
  try {
    await api("/api/scene-updates/rollback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: uniqueId("rollback-update"), result_id: resultId }) });
    notify("Previously accepted scene result restored."); await Promise.all([refreshStatus(true), refreshArtifacts(false)]);
  } catch (error) { notify(error.message, true); }
}

async function refreshStatus(renderContentToo = false) {
  state.status = await api("/api/status");
  renderSceneSwitcher();
  renderNavigation();
  renderSteps();
  const warning = asObject(state.status?.scene_updates?.operator_warning);
  if (warning?.warning_id && warning.warning_id !== state.warningId) {
    state.warningId = warning.warning_id;
    window.alert(`Live operations warning\n\n${warning.message}`);
    await api("/api/live-operations/warning/acknowledge", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ warning_id: warning.warning_id }) }).catch(() => undefined);
  }
  if (renderContentToo || selectedPageId() === "live") renderContent();
}

async function initialize() {
  try {
    const [pagePayload, registryPayload] = await Promise.all([api("/api/pages"), api("/api/scenes")]);
    state.pages = pagePayload.pages;
    state.sceneRegistry = asArray(registryPayload.scenes);
    document.getElementById("scene-select")?.addEventListener("change", event => {
      window.location.assign(`/scenes/${encodeURIComponent(event.target.value)}/pages/setup`);
    });
    window.addEventListener("message", async event => {
      if (event.origin !== window.location.origin || event.data?.type !== "p04-linked-points-exported") return;
      notify(`${event.data.camera_id} linked points exported. Calibration can now be tested if the 4 + 2 point set is complete.`);
      await refreshStatus(true);
    });
    renderNavigation(); await Promise.all([refreshStatus(false), refreshArtifacts(false)]); renderSteps(); renderContent();
    window.setInterval(() => refreshStatus(false), 3000);
  } catch (error) { notify(error.message, true); }
}

function renderSceneSwitcher() {
  const select = document.getElementById("scene-select");
  if (!select) return;
  const current = selectedSceneId();
  const records = state.sceneRegistry.length ? state.sceneRegistry : [{ scene_uuid: current, display_name: state.status.display_name }];
  select.innerHTML = records.map(scene => `<option value="${escapeHtml(scene.scene_uuid)}" ${scene.scene_uuid === current ? "selected" : ""}>${escapeHtml(scene.display_name)} · ${scene.camera_count ?? state.status.camera_roster.length} cameras</option>`).join("");
}

initialize();
