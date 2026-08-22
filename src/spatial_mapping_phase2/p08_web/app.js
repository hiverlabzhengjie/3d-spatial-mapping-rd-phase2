const state = { pages: [], status: null, artifactCatalog: null, selectedCamera: null, artifactSection: "all", deleteImpact: null };

function selectedPageId() {
  const match = window.location.pathname.match(/^\/pages\/([^/]+)$/);
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
    archived: "Archived", missing: "Missing", corrupt: "Changed", deleted: "Deleted permanently"
  })[value] || String(value || "Unknown").replaceAll("_", " ");
}

function uniqueId(prefix) {
  return `${prefix}-${new Date().toISOString().replace(/[-:.TZ]/g, "").toLowerCase()}`;
}

async function api(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
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
  document.getElementById("navigation").innerHTML = state.pages.map(page =>
    `<a class="${page.page_id === selected ? "active" : ""}" href="/pages/${page.page_id}">${escapeHtml(page.title)}</a>`
  ).join("");
  const page = state.pages.find(item => item.page_id === selected) || state.pages[0];
  document.getElementById("page-title").textContent = page?.title || "Workflow";
}

function renderSteps() {
  document.getElementById("step-strip").innerHTML = state.status.operator.steps.map((step, index) => `
    <a href="/pages/${escapeHtml(step.step_id)}" class="step ${escapeHtml(step.state)}">
      <span>${index + 1}</span><div><strong>${escapeHtml(step.title)}</strong><small>${escapeHtml(humanState(step.state))}</small></div>
    </a>`).join("");
}

function statusBadge(value) {
  return `<span class="state ${escapeHtml(value)}">${escapeHtml(humanState(value))}</span>`;
}

function projectOverview() {
  const workflow = state.status.operator;
  const completed = workflow.steps.filter(item => ["complete", "approved"].includes(item.state)).length;
  return `
    <article class="card hero wide"><p class="eyebrow">Current project</p><h2>${escapeHtml(state.status.display_name)}</h2>
      <p>Continue from the next available step. Camera calibration and world poses are ready for all configured cameras.</p>
      <div class="metric-grid"><div class="metric"><strong>${state.status.camera_roster.length}</strong><span>Cameras</span></div>
      <div class="metric"><strong>${completed}/${workflow.steps.length}</strong><span>Steps complete</span></div>
      <div class="metric"><strong>${workflow.inputs_ready ? "Ready" : "Check"}</strong><span>Reconstruction inputs</span></div></div>
      <div class="session-actions"><div><h3>Demonstration run</h3><p>Start over without deleting any captures or generated artifacts. The current run is archived and the processing activity is cleared.</p></div><button id="fresh-session" class="secondary" ${workflow.session.can_start_fresh ? "" : "disabled"}>Start a fresh run</button></div>
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
    <section class="history-section"><div class="history-section-heading"><div><p class="eyebrow">Past versions</p><h2>Return to earlier work</h2><p>Select a workflow step, then choose <strong>Make current</strong>. If later work must be reset, you will see that before anything changes.</p></div></div><div class="history-filters" role="tablist" aria-label="Filter past versions by workflow step">${filters}</div>${pastGroups || `<div class="empty-history"><h3>No past versions in this view</h3><p>Older versions will appear here after another run or selection is created.</p></div>`}</section>
    <section class="history-section cleanup-section"><div><p class="eyebrow">Storage cleanup</p><h2>Remove old files carefully</h2><p>Open <strong>Manage</strong> beside a past version to check, archive, or permanently delete it. Current files, accepted inputs, authority records, rollback records and files needed by another retained result are protected.</p></div><div class="storage-metrics"><div><strong>${storage.past_version_count || 0}</strong><span>Past versions</span></div><div><strong>${escapeHtml(formatBytes(storage.past_retained_byte_count || 0))}</strong><span>Past files retained</span></div><div><strong>${storage.protected_version_count || 0}</strong><span>Protected records</span></div><div><strong>${storage.deleted_version_count || 0}</strong><span>Deleted records</span></div></div><div class="danger-note"><strong>Permanent means permanent.</strong> Only the exact displayed file is removed. The console keeps its name, hash and deletion record, but cannot restore the file.</div></section>
    <details class="recent-changes"><summary>Recent history actions</summary><div class="history-list">${recentEvents || `<p class="muted">No management actions recorded yet.</p>`}</div></details>
    ${deletionDialog()}
  </section>`;
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
  return `<article class="card wide"><h2>Configured cameras</h2><div class="camera-grid">${state.status.camera_roster.map(camera => `
    <div class="camera-tile"><span class="camera-icon">●</span><div><strong>${escapeHtml(camera.display_name)}</strong><small>${camera.enabled ? "Connected to this workflow" : "Disabled"}</small></div></div>`).join("")}</div></article>`;
}

function embeddedTool(page, emptyMessage) {
  if (!page.tool_url) return `<article class="card wide"><h2>Tool unavailable</h2><p>${escapeHtml(emptyMessage)}</p></article>`;
  return `<article class="card wide tool-card"><iframe class="tool-frame" src="${escapeHtml(page.tool_url)}" title="${escapeHtml(page.title)}"></iframe></article>`;
}

function calibrationPage(page) {
  const cameras = state.status.operator.cameras;
  const tools = page.calibration_tools || [];
  if (!state.selectedCamera) state.selectedCamera = cameras[0]?.camera_id || tools[0]?.camera_id;
  const selected = cameras.find(camera => camera.camera_id === state.selectedCamera) || cameras[0];
  const selectedTool = tools.find(tool => tool.camera_id === state.selectedCamera);
  return `<article class="card wide"><div class="section-heading"><div><h2>Camera correspondence workspace</h2><p>Switch cameras to inspect or refine image correspondences and the resulting world pose in one place.</p></div>${selected ? statusBadge(selected.ready ? "complete" : "attention") : ""}</div>
    <div class="camera-tabs">${state.status.camera_roster.map(camera => `<button type="button" class="camera-tab ${camera.camera_id === state.selectedCamera ? "active" : ""}" data-camera="${escapeHtml(camera.camera_id)}">${escapeHtml(camera.display_name)}</button>`).join("")}</div>
    ${selected ? cameraSummary(selected, false) : `<div class="warning">Camera results are unavailable. Rework this camera before reconstruction.</div>`}
  </article>
  <article class="card wide tool-card">${selectedTool ? `<iframe class="tool-frame" src="${escapeHtml(selectedTool.tool_url)}" title="Calibration correspondence for ${escapeHtml(state.selectedCamera)}"></iframe>` : `<div class="empty"><h2>No correspondence workspace configured</h2><p>The final camera result is shown above, but the editing workspace is not mounted for this camera.</p></div>`}</article>`;
}

function cameraSummary(camera, expanded = true) {
  const i = camera.intrinsics;
  const p = camera.pose;
  const rows = p.matrix.map(row => `<tr>${row.map(value => `<td>${Number(value).toFixed(6)}</td>`).join("")}</tr>`).join("");
  return `<div class="camera-summary ${expanded ? "expanded" : ""}">
    <div><h3>Intrinsics</h3><dl class="details"><dt>Model</dt><dd>${escapeHtml(i.model)}</dd><dt>Resolution</dt><dd>${i.resolution[0]} × ${i.resolution[1]} px</dd><dt>Focal length</dt><dd>fx ${i.fx_pixels.toFixed(6)} · fy ${i.fy_pixels.toFixed(6)} px</dd><dt>Principal point</dt><dd>cx ${i.cx_pixels.toFixed(6)} · cy ${i.cy_pixels.toFixed(6)} px</dd><dt>Distortion</dt><dd>${i.distortion.map(value => Number(value).toFixed(9)).join(", ")}</dd></dl></div>
    <div><h3>6 DoF pose in world frame</h3><dl class="details"><dt>Position (m)</dt><dd>x ${p.position_metres[0].toFixed(6)} · y ${p.position_metres[1].toFixed(6)} · z ${p.position_metres[2].toFixed(6)}</dd><dt>Orientation</dt><dd>yaw ${p.orientation_zyx_degrees.yaw.toFixed(4)}° · pitch ${p.orientation_zyx_degrees.pitch.toFixed(4)}° · roll ${p.orientation_zyx_degrees.roll.toFixed(4)}°</dd></dl><details ${expanded ? "" : ""}><summary>Exact T_world_from_camera matrix</summary><table class="matrix"><tbody>${rows}</tbody></table></details></div>
  </div>`;
}

function reconstructionPage() {
  const workflow = state.status.operator;
  const geometry = workflow.geometry;
  const reconstructing = workflow.active_action?.action === "all-camera-static-reconstruction";
  return `<article class="card wide"><div class="section-heading"><div><h2>Camera inputs</h2><p>Static reconstruction uses these exact accepted intrinsics and world-frame poses.</p></div>${statusBadge(workflow.inputs_ready ? "complete" : "attention")}</div>
    ${workflow.input_issues.length ? `<div class="warning"><strong>Reconstruction is blocked.</strong> ${workflow.input_issues.map(escapeHtml).join(" ")} Return to Calibration & pose and rework the affected camera.</div>` : ""}
    <div class="accordion-list">${workflow.cameras.map(camera => `<details><summary>${escapeHtml(camera.camera_id)} ${statusBadge(camera.ready ? "complete" : "attention")}</summary>${cameraSummary(camera)}</details>`).join("")}</div>
  </article>
  ${reconstructing ? `<article class="card wide progress-card"><strong>Static reconstruction is running.</strong><p>DA3 inference, point-cloud export and verification normally take about one to two minutes. Preview will become available only after all three complete.</p></article>` : ""}
  <article class="card action-card"><h2>Build combined point cloud</h2><p>Run pose-conditioned static reconstruction for every ready camera, then export and verify the combined geometry.</p><button id="run-reconstruction" ${geometry.can_run ? "" : "disabled"}>${reconstructing ? "Static reconstruction running…" : "Run static reconstruction"}</button></article>
  <article class="card action-card"><h2>Geometry review</h2><p>Open the current combined point cloud in Rerun. Approve it only after visual inspection.</p><div class="button-row"><button id="open-geometry" ${geometry.can_preview ? "" : "disabled"}>Open Rerun preview</button><button id="approve-geometry" class="secondary" ${geometry.previewed && !geometry.approved ? "" : "disabled"}>${geometry.approved ? "Geometry approved" : "Approve geometry"}</button></div></article>
  ${jobsCard()}`;
}

function floorPage() {
  const floor = state.status.operator.floor;
  const active = state.status.operator.active_action;
  const floorRunning = active?.action === "floor-completion";
  const previewRunning = active?.action === "build-and-open-floor-preview";
  return `<article class="card wide"><div class="section-heading"><div><h2>Authoritative floor at Z=0</h2><p>Create the deterministic floor derivative only after the combined geometry is approved. Original coloured point-cloud samples remain unchanged.</p></div>${statusBadge(floor.approved ? "approved" : floor.can_generate ? "ready" : "blocked")}</div>
    ${!floor.can_generate && !floor.approved ? `<div class="warning">Approve the combined geometry in Static reconstruction before generating the floor result.</div>` : ""}
  </article>
  ${floorRunning || previewRunning ? `<article class="card wide progress-card"><strong>${floorRunning ? "Floor generation" : "Final preview"} is running.</strong><p>The current action is protected from duplicate clicks. Existing immutable results are not overwritten.</p></article>` : ""}
  <article class="card action-card"><h2>Generate final point cloud</h2><p>Add the configured authoritative floor plane and produce a new immutable result.</p><button id="run-floor" ${floor.can_generate ? "" : "disabled"}>${floorRunning ? "Generating floor-completed result…" : "Generate floor-completed result"}</button></article>
  <article class="card action-card"><h2>Final review</h2><p>Build and open the newest result, or inspect the current final recording. Approve only after checking it in Rerun.</p><div class="button-stack"><button id="build-floor-preview" ${floor.can_build_preview ? "" : "disabled"} data-floor-job="${escapeHtml(floor.current_floor_job_id || "")}">${previewRunning ? "Building final preview…" : "Build & open newest preview"}</button><button id="open-floor" class="secondary" ${floor.can_preview ? "" : "disabled"}>${floor.opening_latest_approved ? "Open latest approved final preview" : "Open current final preview"}</button><button id="approve-floor" class="secondary" ${floor.previewed && !floor.approved ? "" : "disabled"}>${floor.approved ? "Final result approved" : "Approve final result"}</button></div></article>
  ${jobsCard()}`;
}

function resultsPage() {
  const workflow = state.status.operator;
  return `<article class="card hero wide"><p class="eyebrow">Workflow result</p><h2>${workflow.floor.approved ? "Final result approved" : "Final review is still required"}</h2><p>${workflow.floor.approved ? "The current floor-completed point cloud has been previewed and approved in this console session." : "Open the final Rerun preview on the Floor refinement page and approve it when it meets your needs."}</p><a class="button-link" href="/pages/floor">Go to floor review</a></article>
    ${cameraRoster()}`;
}

function jobsCard() {
  const jobs = state.status.jobs;
  if (!jobs.length) return `<article class="card wide"><h2>Recent activity</h2><p class="muted">No processing actions have run in this session.</p></article>`;
  const labels = { "all-camera-static-reconstruction": "Static reconstruction", "floor-completion": "Floor generation", "build-and-open-floor-preview": "Final preview" };
  return `<article class="card wide"><h2>Recent activity</h2><div class="job-list">${jobs.slice().reverse().map(job => `<div class="job-row"><div><strong>${escapeHtml(labels[job.action] || "Workflow action")}</strong><small>${escapeHtml(job.error_message || "")}</small></div>${statusBadge(job.state)}</div>`).join("")}</div></article>`;
}

function renderContent() {
  const page = state.pages.find(item => item.page_id === selectedPageId()) || state.pages[0];
  let html = "";
  if (page.page_id === "setup") html = projectOverview();
  if (page.page_id === "artifacts") html = artifactVersionsPage();
  if (page.page_id === "facility") html = embeddedTool(page, "Facility registration is not configured in this launch.");
  if (page.page_id === "capture") html = embeddedTool(page, "Live capture is not enabled in this launch.");
  if (page.page_id === "calibration") html = calibrationPage(page);
  if (page.page_id === "reconstruction") html = reconstructionPage();
  if (page.page_id === "floor") html = floorPage();
  if (page.page_id === "results") html = resultsPage();
  document.getElementById("content").innerHTML = html;
  bindActions();
}

function bindActions() {
  document.querySelectorAll(".camera-tab").forEach(button => button.addEventListener("click", () => { state.selectedCamera = button.dataset.camera; renderContent(); }));
  document.querySelectorAll(".history-filter").forEach(button => button.addEventListener("click", () => { state.artifactSection = button.dataset.section; renderContent(); }));
  document.getElementById("run-reconstruction")?.addEventListener("click", () => startJob("/api/jobs/reconstruction", { job_id: uniqueId("reconstruction") }, "Static reconstruction"));
  document.getElementById("run-floor")?.addEventListener("click", () => startJob("/api/jobs/floor", { job_id: uniqueId("floor") }, "Floor generation"));
  document.getElementById("build-floor-preview")?.addEventListener("click", event => startJob("/api/jobs/floor-preview", { job_id: uniqueId("floor-preview"), floor_job_id: event.currentTarget.dataset.floorJob }, "Final preview"));
  document.getElementById("open-geometry")?.addEventListener("click", () => openPreview("geometry"));
  document.getElementById("open-floor")?.addEventListener("click", () => openPreview("floor"));
  document.getElementById("approve-geometry")?.addEventListener("click", () => approve("geometry"));
  document.getElementById("approve-floor")?.addEventListener("click", () => approve("floor"));
  document.getElementById("fresh-session")?.addEventListener("click", startFreshSession);
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

async function refreshArtifacts(render = false) {
  state.artifactCatalog = await api("/api/artifacts");
  if (render && selectedPageId() === "artifacts") renderContent();
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

async function refreshStatus(renderContentToo = false) {
  state.status = await api("/api/status");
  document.getElementById("scene-pill").textContent = `${state.status.display_name} · ${state.status.camera_roster.length} cameras`;
  renderSteps();
  if (renderContentToo) renderContent();
}

async function initialize() {
  try {
    state.pages = (await api("/api/pages")).pages;
    renderNavigation(); await Promise.all([refreshStatus(false), refreshArtifacts(false)]); renderSteps(); renderContent();
    window.setInterval(() => refreshStatus(false), 3000);
  } catch (error) { notify(error.message, true); }
}

initialize();
