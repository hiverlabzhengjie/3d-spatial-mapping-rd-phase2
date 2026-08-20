const state = { pages: [], status: null };

function selectedPageId() {
  const match = window.location.pathname.match(/^\/pages\/([^/]+)$/);
  return match ? match[1] : "setup";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);
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

function renderPhases() {
  document.getElementById("phase-strip").innerHTML = state.status.phases.map(phase => `
    <div class="phase-chip" title="${escapeHtml(phase.message)}">
      <strong>${phase.phase_id}</strong><span class="state ${phase.state}">${escapeHtml(phase.state)}</span>
    </div>`).join("");
}

function phaseTable(phaseIds) {
  const selected = state.status.phases.filter(phase => phaseIds.includes(phase.phase_id));
  return `<table><thead><tr><th>Phase</th><th>State</th><th>Operator message</th></tr></thead><tbody>${selected.map(phase => `
    <tr><td><strong>${phase.phase_id}</strong></td><td><span class="state ${phase.state}">${escapeHtml(phase.state)}</span></td>
    <td>${escapeHtml(phase.message)}${phase.reasons.map(reason => `<div class="reason">${escapeHtml(reason)}</div>`).join("")}</td></tr>`).join("")}</tbody></table>`;
}

function rosterCard() {
  return `<article class="card"><h2>Configured camera roster</h2><table><thead><tr><th>Camera</th><th>Endpoint binding</th><th>Enabled</th></tr></thead><tbody>${state.status.camera_roster.map(camera => `
    <tr><td>${escapeHtml(camera.display_name)}<br><code>${escapeHtml(camera.camera_id)}</code></td><td><code>${escapeHtml(camera.endpoint_environment_key || "not configured")}</code></td><td>${camera.enabled ? "yes" : "no"}</td></tr>`).join("")}</tbody></table></article>`;
}

function artifactCard() {
  return `<article class="card wide"><h2>Immutable artifacts & authority</h2><table><thead><tr><th>Phase</th><th>Artifact</th><th>Authority</th><th>Selected</th></tr></thead><tbody>${state.status.artifacts.map(artifact => `
    <tr><td>${artifact.phase_id}</td><td><code>${escapeHtml(artifact.artifact_id)}</code><br><span class="muted">${escapeHtml(artifact.kind)}</span></td><td>${escapeHtml(artifact.authority)}</td><td>${artifact.selected ? "yes" : "no"}</td></tr>`).join("")}</tbody></table></article>`;
}

function jobsCard() {
  return `<article class="card wide"><h2>Bounded jobs</h2>${state.status.jobs.length ? `<table><thead><tr><th>Job</th><th>Phase</th><th>Action</th><th>State</th><th>Failure</th></tr></thead><tbody>${state.status.jobs.map(job => `
    <tr><td><code>${escapeHtml(job.job_id)}</code></td><td>${job.phase_id}</td><td>${escapeHtml(job.action)}</td><td><span class="state ${job.state}">${job.state}</span></td><td>${escapeHtml(job.error_message || "—")}</td></tr>`).join("")}</tbody></table>` : `<p class="muted">No jobs have run in this console process.</p>`}</article>`;
}

function toolCard(page) {
  if (!page.tools?.length) return `<article class="card wide"><h2>Existing stage tool</h2><p class="muted">This adapter is not configured for the current launcher. The stage status and immutable references remain available above.</p></article>`;
  return page.tools.map(tool => `<article class="card wide"><h2>${escapeHtml(tool.tool_id.toUpperCase())} authoritative stage service</h2><p class="muted">Embedded through an adapter; the standalone launcher remains available.</p><iframe class="tool-frame" src="${tool.tool_url}" title="${escapeHtml(tool.tool_id.toUpperCase())} tool"></iframe></article>`).join("");
}

function geometryActions() {
  return `<article class="card"><h2>Floor processing</h2><p class="muted">Creates a new immutable derivative; P07 v2 is never overwritten.</p><form id="floor-form"><label>Job ID<input name="job_id" pattern="[a-z0-9][a-z0-9._-]{0,63}" required value="p08-floor-manual"></label><button type="submit">Start bounded job</button></form></article>`;
}

function rerunActions() {
  const recordings = state.status.artifacts.filter(item => item.kind === "rerun-recording" && item.selected);
  return `<article class="card"><h2>Safe Rerun inspection</h2><p class="muted">Only a selected hash-verified recording beneath the configured artifact root can launch.</p><form id="rerun-form"><label>Action ID<input name="action_id" pattern="[a-z0-9][a-z0-9._-]{0,63}" required value="launch-rerun-manual"></label><label>Selected recording<select name="artifact_id" required>${recordings.map(item => `<option value="${escapeHtml(item.artifact_id)}">${escapeHtml(item.artifact_id)}</option>`).join("")}</select></label><button type="submit">Open selected Rerun</button></form></article>`;
}

function renderContent() {
  const page = state.pages.find(item => item.page_id === selectedPageId()) || state.pages[0];
  let html = `<article class="card wide"><h2>Phase status</h2>${phaseTable(page.phase_ids)}</article>`;
  if (page.page_id === "setup") html += rosterCard() + artifactCard();
  if (["facility", "capture", "calibration"].includes(page.page_id)) html += toolCard(page);
  if (page.page_id === "reconstruction") html += artifactCard() + jobsCard();
  if (page.page_id === "geometry") html += geometryActions() + rerunActions() + jobsCard();
  if (page.page_id === "artifacts") html += artifactCard() + rerunActions();
  document.getElementById("content").innerHTML = html;
  document.getElementById("floor-form")?.addEventListener("submit", submitFloor);
  document.getElementById("rerun-form")?.addEventListener("submit", launchRerun);
}

async function submitFloor(event) {
  event.preventDefault();
  const jobId = new FormData(event.currentTarget).get("job_id");
  try {
    await api("/api/jobs/floor", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ job_id: jobId }) });
    notify(`Floor job ${jobId} queued.`); await refreshStatus(true); monitorJob(jobId);
  } catch (error) { notify(error.message, true); }
}

async function monitorJob(jobId) {
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (["complete", "failed", "cancelled"].includes(job.state)) {
      notify(`Floor job ${jobId}: ${job.state}.`, job.state === "failed");
      await refreshStatus(true);
      return;
    }
    window.setTimeout(() => monitorJob(jobId), 750);
  } catch (error) { notify(error.message, true); }
}

async function launchRerun(event) {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  try {
    const result = await api("/api/rerun/launch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action_id: data.get("action_id"), artifact_id: data.get("artifact_id") }) });
    notify(`Launched ${result.artifact_id}.`);
  } catch (error) { notify(error.message, true); }
}

async function refreshStatus(renderContentToo = false) {
  state.status = await api("/api/status");
  document.getElementById("scene-pill").textContent = `${state.status.display_name} · ${state.status.camera_roster.length} cameras`;
  renderPhases();
  if (renderContentToo) renderContent();
}

async function initialize() {
  try {
    state.pages = (await api("/api/pages")).pages;
    renderNavigation(); await refreshStatus(true);
    window.setInterval(refreshStatus, 3000);
  } catch (error) { notify(error.message, true); }
}

initialize();
