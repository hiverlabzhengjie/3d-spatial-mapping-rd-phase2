"use strict";

const ui = {};
let state = null;
let derived = null;
let interaction = null;
let draftImagePoint = null;
let draftPlanPoint = null;
let frameZoom = 1;
let planZoom = 1;

document.addEventListener("DOMContentLoaded", async () => {
  [
    "notReady", "workspace", "revisionBadge", "exportButton", "validationExportButton", "cameraId", "captureCameraId", "approvedFrame",
    "solveCount", "heldOutCount", "validationCount", "toggleImportButton", "importForm", "sourcePath",
    "captureDelay", "captureButton", "captureStatus", "newFrameId", "profileVersion",
    "expectedSha", "frameCards", "imagePointReadout",
    "worldXYReadout", "pickImageButton", "pickPlanButton", "frameViewport", "frameScene",
    "frameImage", "frameOverlay", "noApprovedFrame", "planViewport", "planScene", "planImage",
    "planOverlay", "frameZoomOut", "frameZoomValue", "frameZoomIn", "frameZoomFit",
    "planZoomOut", "planZoomValue", "planZoomIn", "planZoomFit", "landmarkForm",
    "landmarkId", "landmarkName", "physicalMeaning", "zMetres",
    "zSource", "zUncertainty", "landmarkRole", "saveLandmarkButton", "draftState",
    "landmarkTable", "previewModal", "previewTitle", "previewImage", "previewDetails",
    "closePreviewButton", "toast",
  ].forEach((id) => { ui[id] = document.getElementById(id); });
  wireEvents();
  await loadState();
});

function wireEvents() {
  ui.toggleImportButton.addEventListener("click", () => {
    ui.importForm.hidden = !ui.importForm.hidden;
  });
  ui.importForm.addEventListener("submit", importFrame);
  ui.captureButton.addEventListener("click", captureTimedCandidate);
  ui.pickImageButton.addEventListener("click", () => setInteraction("image"));
  ui.pickPlanButton.addEventListener("click", () => setInteraction("plan"));
  ui.frameOverlay.addEventListener("click", (event) => selectPoint(event, "image"));
  ui.planOverlay.addEventListener("click", (event) => selectPoint(event, "plan"));
  ui.frameOverlay.addEventListener("pointermove", (event) => showLivePoint(event, "image"));
  ui.planOverlay.addEventListener("pointermove", (event) => showLivePoint(event, "plan"));
  ui.frameOverlay.addEventListener("pointerleave", () => restoreCoordinateReadout("image"));
  ui.planOverlay.addEventListener("pointerleave", () => restoreCoordinateReadout("plan"));
  ui.frameZoomIn.addEventListener("click", () => setPaneZoom("frame", frameZoom + .15));
  ui.frameZoomOut.addEventListener("click", () => setPaneZoom("frame", frameZoom - .15));
  ui.frameZoomFit.addEventListener("click", () => fitPane("frame"));
  ui.planZoomIn.addEventListener("click", () => setPaneZoom("plan", planZoom + .15));
  ui.planZoomOut.addEventListener("click", () => setPaneZoom("plan", planZoom - .15));
  ui.planZoomFit.addEventListener("click", () => fitPane("plan"));
  ui.closePreviewButton.addEventListener("click", closePreview);
  ui.previewModal.addEventListener("click", (event) => {
    if (event.target === ui.previewModal) closePreview();
  });
  ui.landmarkForm.addEventListener("submit", saveLandmark);
  ui.exportButton.addEventListener("click", exportSnapshot);
  ui.validationExportButton.addEventListener("click", exportD034Validation);
}

async function loadState() {
  try {
    const payload = await api("/api/status");
    if (!payload.has_state) {
      ui.notReady.hidden = false;
      ui.workspace.hidden = true;
      ui.exportButton.disabled = true;
      ui.validationExportButton.disabled = true;
      return;
    }
    setState(payload.state);
  } catch (error) {
    toast(error.message, true);
  }
}

function setState(payload) {
  state = payload;
  derived = payload.derived;
  delete state.derived;
  ui.notReady.hidden = true;
  ui.workspace.hidden = false;
  ui.exportButton.disabled = false;
  ui.validationExportButton.disabled = false;
  const pointCount = state.landmarks.length;
  ui.revisionBadge.textContent = `${pointCount} linked ${pointCount === 1 ? "point" : "points"}`;
  ui.revisionBadge.title = `Workspace history version ${state.revision}`;
  ui.cameraId.textContent = state.camera_id;
  ui.captureCameraId.textContent = state.camera_id;
  ui.approvedFrame.textContent = derived.approved_frame_id || "None";
  ui.solveCount.textContent = String(derived.solve_count);
  ui.heldOutCount.textContent = String(derived.held_out_count);
  ui.validationCount.textContent = String(derived.d034_validation_count);
  configurePlan();
  configureApprovedFrame();
  renderFrames();
  renderLandmarks();
  renderDraft();
}

function configurePlan() {
  const reference = state.facility_reference;
  ui.planScene.style.width = `${reference.plan_image_width_pixels}px`;
  ui.planScene.style.height = `${reference.plan_image_height_pixels}px`;
  ui.planOverlay.setAttribute("viewBox", `0 0 ${reference.plan_image_width_pixels} ${reference.plan_image_height_pixels}`);
  ui.planImage.src = `/api/plan-image?v=${reference.plan_image_sha256}`;
  ui.planImage.onload = () => fitPane("plan");
}

function configureApprovedFrame() {
  const frame = approvedFrame();
  if (!frame) {
    ui.frameImage.removeAttribute("src");
    ui.frameImage.hidden = true;
    ui.frameScene.style.width = "100%";
    ui.frameScene.style.height = "100%";
    ui.noApprovedFrame.hidden = false;
    ui.pickImageButton.disabled = true;
    setPaneZoom("frame", 1);
    return;
  }
  ui.noApprovedFrame.hidden = true;
  ui.frameImage.hidden = false;
  ui.pickImageButton.disabled = false;
  ui.frameScene.style.width = `${frame.image_width_pixels}px`;
  ui.frameScene.style.height = `${frame.image_height_pixels}px`;
  ui.frameOverlay.setAttribute("viewBox", `0 0 ${frame.image_width_pixels} ${frame.image_height_pixels}`);
  ui.frameImage.src = `/api/frames/${encodeURIComponent(frame.frame_id)}/image?v=${frame.sha256}`;
  ui.frameImage.onload = () => fitPane("frame");
}

function renderFrames() {
  ui.frameCards.replaceChildren();
  if (!state.frames.length) {
    const empty = document.createElement("p");
    empty.textContent = `No candidate frames yet. Start a timed ${state.camera_id} capture when the scene is ready, or add a local frame.`;
    ui.frameCards.append(empty);
    return;
  }
  state.frames.forEach((frame) => {
    const card = document.createElement("article");
    card.className = `frame-card ${frame.status}`;
    const title = document.createElement("strong");
    title.textContent = frame.frame_id;
    const status = document.createElement("span");
    status.className = "badge neutral";
    status.textContent = frame.status;
    const details = document.createElement("p");
    details.textContent = `${frame.image_width_pixels}×${frame.image_height_pixels} · ${frame.profile_version} · ${frame.capture_kind}`;
    const thumbnail = document.createElement("img");
    thumbnail.className = "frame-thumbnail";
    thumbnail.alt = `Preview of ${frame.frame_id}`;
    thumbnail.src = `/api/frames/${encodeURIComponent(frame.frame_id)}/image?v=${frame.sha256}`;
    thumbnail.addEventListener("click", () => openPreview(frame));
    const hash = document.createElement("code");
    hash.textContent = frame.sha256;
    const note = document.createElement("p");
    note.textContent = frame.review_note || "Not reviewed yet";
    const actions = document.createElement("div");
    actions.className = "actions";
    const preview = document.createElement("button");
    preview.textContent = "Open preview";
    preview.addEventListener("click", () => openPreview(frame));
    const approve = document.createElement("button");
    approve.textContent = frame.status === "approved" ? "Approved" : "Approve";
    approve.disabled = frame.status === "approved";
    approve.addEventListener("click", () => reviewFrame(frame.frame_id, "approved"));
    const reject = document.createElement("button");
    reject.textContent = "Reject / recapture";
    reject.disabled = frame.status === "rejected";
    reject.addEventListener("click", () => reviewFrame(frame.frame_id, "rejected"));
    actions.append(preview, approve, reject);
    card.append(title, status, details, thumbnail, hash, note, actions);
    ui.frameCards.append(card);
  });
}

async function captureTimedCandidate() {
  const delay = Number(ui.captureDelay.value);
  ui.captureButton.disabled = true;
  let remaining = delay;
  ui.captureStatus.textContent = remaining > 0 ? `Capturing in ${remaining}s` : "Capturing now";
  const timer = setInterval(() => {
    remaining -= 1;
    ui.captureStatus.textContent = remaining > 0 ? `Capturing in ${remaining}s` : "Connecting…";
  }, 1000);
  try {
    const payload = await api("/api/capture-candidate", {
      method: "POST",
      body: JSON.stringify({ delay_seconds: delay }),
    });
    clearInterval(timer);
    setState(payload);
    const newest = state.frames[state.frames.length - 1];
    ui.captureStatus.textContent = "Candidate ready";
    openPreview(newest);
    toast(`Timed ${state.camera_id} candidate captured. Review before approving or recapturing.`);
  } catch (error) {
    clearInterval(timer);
    ui.captureStatus.textContent = "Capture failed";
    toast(error.message, true);
  } finally {
    ui.captureButton.disabled = false;
  }
}

function openPreview(frame) {
  ui.previewTitle.textContent = frame.frame_id;
  ui.previewImage.src = `/api/frames/${encodeURIComponent(frame.frame_id)}/image?v=${frame.sha256}`;
  const timing = frame.observed_at_utc ? ` · ${frame.observed_at_utc}` : "";
  ui.previewDetails.textContent = `${frame.image_width_pixels}×${frame.image_height_pixels} · ${frame.capture_kind}${timing} · SHA-256 ${frame.sha256}`;
  ui.previewModal.hidden = false;
}

function closePreview() {
  ui.previewModal.hidden = true;
  ui.previewImage.removeAttribute("src");
}

async function importFrame(event) {
  event.preventDefault();
  try {
    const payload = await api("/api/frames", {
      method: "POST",
      body: JSON.stringify({
        source_path: ui.sourcePath.value.trim(),
        frame_id: ui.newFrameId.value.trim(),
        profile_version: ui.profileVersion.value.trim(),
        expected_sha256: ui.expectedSha.value.trim() || null,
      }),
    });
    setState(payload);
    ui.sourcePath.value = "";
    ui.newFrameId.value = "";
    ui.expectedSha.value = "";
    toast("Candidate frame imported with immutable identity.");
  } catch (error) {
    toast(error.message, true);
  }
}

async function reviewFrame(frameId, status) {
  const defaultNote = status === "approved" ? "approved for primary pose annotation" : "recapture requested";
  const note = prompt(status === "approved" ? "Approval note (optional)" : "Why should this frame be recaptured?", defaultNote);
  if (note === null) return;
  try {
    const payload = await api(`/api/frames/${encodeURIComponent(frameId)}/review`, {
      method: "PUT",
      body: JSON.stringify({ status, note: note.trim() || null }),
    });
    clearDraft();
    setState(payload);
    toast(status === "approved" ? "Primary frame approved." : "Frame rejected; evidence retained.");
  } catch (error) {
    toast(error.message, true);
  }
}

function setInteraction(kind) {
  if (kind === "image" && !approvedFrame()) return toast("Approve a primary frame first.", true);
  interaction = kind;
  ui.pickImageButton.classList.toggle("active", kind === "image");
  ui.pickPlanButton.classList.toggle("active", kind === "plan");
  toast(kind === "image" ? "Click the exact feature in the approved frame." : "Click the same feature on the plan.");
}

function selectPoint(event, kind) {
  if (interaction !== kind) return;
  const overlay = kind === "image" ? ui.frameOverlay : ui.planOverlay;
  const point = pointFromEvent(event, overlay);
  if (kind === "image") draftImagePoint = point;
  else draftPlanPoint = point;
  interaction = null;
  ui.pickImageButton.classList.remove("active");
  ui.pickPlanButton.classList.remove("active");
  renderDraft();
}

function showLivePoint(event, kind) {
  if (kind === "image" && !approvedFrame()) return;
  const overlay = kind === "image" ? ui.frameOverlay : ui.planOverlay;
  const point = pointFromEvent(event, overlay);
  if (kind === "image") {
    ui.imagePointReadout.textContent = `live u ${point.u.toFixed(1)} · v ${point.v.toFixed(1)}`;
    return;
  }
  const xy = worldXY(point);
  ui.worldXYReadout.textContent = `live X ${xy.x_metres.toFixed(3)} m · Y ${xy.y_metres.toFixed(3)} m · plan ${point.u.toFixed(1)}, ${point.v.toFixed(1)}`;
}

function restoreCoordinateReadout(kind) {
  if (kind === "image") {
    ui.imagePointReadout.textContent = draftImagePoint ? `u ${draftImagePoint.u.toFixed(1)} · v ${draftImagePoint.v.toFixed(1)}` : "u — · v —";
    return;
  }
  const xy = draftPlanPoint ? worldXY(draftPlanPoint) : null;
  ui.worldXYReadout.textContent = xy ? `X ${xy.x_metres.toFixed(3)} m · Y ${xy.y_metres.toFixed(3)} m` : "X — · Y —";
}

function setPaneZoom(kind, value) {
  const zoom = clamp(value, .15, 4);
  const scene = kind === "frame" ? ui.frameScene : ui.planScene;
  const dimensions = paneDimensions(kind);
  if (!dimensions) return;
  scene.style.transform = `scale(${zoom})`;
  scene.style.marginRight = `${dimensions.width * (zoom - 1)}px`;
  scene.style.marginBottom = `${dimensions.height * (zoom - 1)}px`;
  if (kind === "frame") {
    frameZoom = zoom;
    ui.frameZoomValue.textContent = `${Math.round(zoom * 100)}%`;
  } else {
    planZoom = zoom;
    ui.planZoomValue.textContent = `${Math.round(zoom * 100)}%`;
  }
}

function fitPane(kind) {
  const dimensions = paneDimensions(kind);
  if (!dimensions) return;
  const viewport = kind === "frame" ? ui.frameViewport : ui.planViewport;
  const fit = Math.min(
    (viewport.clientWidth - 24) / dimensions.width,
    (viewport.clientHeight - 24) / dimensions.height,
    1,
  );
  setPaneZoom(kind, fit);
  viewport.scrollTo(0, 0);
}

function paneDimensions(kind) {
  if (kind === "frame") {
    const frame = approvedFrame();
    return frame ? { width: frame.image_width_pixels, height: frame.image_height_pixels } : null;
  }
  return {
    width: state.facility_reference.plan_image_width_pixels,
    height: state.facility_reference.plan_image_height_pixels,
  };
}

function pointFromEvent(event, overlay) {
  const point = overlay.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const local = point.matrixTransform(overlay.getScreenCTM().inverse());
  const box = overlay.viewBox.baseVal;
  return { u: clamp(local.x, 0, box.width), v: clamp(local.y, 0, box.height) };
}

function renderDraft() {
  ui.imagePointReadout.textContent = draftImagePoint ? `u ${draftImagePoint.u.toFixed(1)} · v ${draftImagePoint.v.toFixed(1)}` : "u — · v —";
  const xy = draftPlanPoint ? worldXY(draftPlanPoint) : null;
  ui.worldXYReadout.textContent = xy ? `X ${xy.x_metres.toFixed(3)} m · Y ${xy.y_metres.toFixed(3)} m` : "X — · Y —";
  ui.draftState.textContent = draftImagePoint && draftPlanPoint ? "Ready for measured Z" : draftImagePoint || draftPlanPoint ? "One click remaining" : "Waiting for two clicks";
  renderOverlays();
}

function renderOverlays() {
  ui.frameOverlay.replaceChildren();
  ui.planOverlay.replaceChildren();
  const approved = approvedFrame();
  state.landmarks.forEach((landmark) => {
    if (approved && landmark.frame_id === approved.frame_id) addMarker(ui.frameOverlay, landmark.image_point, landmark.landmark_id, landmark.role);
    addMarker(ui.planOverlay, landmark.plan_point, landmark.landmark_id, landmark.role);
  });
  if (draftImagePoint) addMarker(ui.frameOverlay, draftImagePoint, "draft", "draft");
  if (draftPlanPoint) addMarker(ui.planOverlay, draftPlanPoint, "draft", "draft");
}

function addMarker(overlay, point, label, role) {
  const circle = svg("circle", { cx: point.u, cy: point.v, r: 12, class: role === "solve" ? "marker-solve" : role === "held-out" ? "marker-held" : role === "d034-validation" ? "marker-validation" : "marker-draft" });
  const text = svg("text", { x: point.u + 17, y: point.v - 14, class: "marker-label" });
  text.textContent = label;
  overlay.append(circle, text);
}

async function saveLandmark(event) {
  event.preventDefault();
  const frame = approvedFrame();
  if (!frame) return toast("Approve a primary frame first.", true);
  if (!draftImagePoint || !draftPlanPoint) return toast("Pick both the image pixel and plan point.", true);
  try {
    const payload = await api("/api/landmarks", {
      method: "POST",
      body: JSON.stringify({
        landmark_id: ui.landmarkId.value.trim(),
        name: ui.landmarkName.value.trim(),
        physical_meaning: ui.physicalMeaning.value.trim(),
        frame_id: frame.frame_id,
        image_point: draftImagePoint,
        plan_point: draftPlanPoint,
        z_metres: Number(ui.zMetres.value),
        z_source: ui.zSource.value.trim() || null,
        z_uncertainty_metres: ui.zUncertainty.value === "" ? null : Number(ui.zUncertainty.value),
        role: ui.landmarkRole.value,
      }),
    });
    clearDraft();
    ui.landmarkForm.reset();
    ui.landmarkRole.value = "solve";
    setState(payload);
    toast("Linked (u, v) ↔ (X, Y, Z) correspondence saved.");
  } catch (error) {
    toast(error.message, true);
  }
}

function renderLandmarks() {
  ui.landmarkTable.replaceChildren();
  const header = row(["Landmark", "Role", "Image (u, v)", "Plan-derived (X, Y)", "Measured Z", ""]);
  header.classList.add("header");
  ui.landmarkTable.append(header);
  state.landmarks.forEach((landmark) => {
    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.addEventListener("click", () => removeLandmark(landmark.landmark_id));
    const item = document.createElement("div");
    item.append(document.createElement("strong"), document.createElement("span"));
    item.children[0].textContent = landmark.name;
    item.children[1].textContent = landmark.landmark_id;
    const record = row([
      item,
      landmark.role,
      `${landmark.image_point.u.toFixed(1)}, ${landmark.image_point.v.toFixed(1)}`,
      `${landmark.world_point.x_metres.toFixed(3)}, ${landmark.world_point.y_metres.toFixed(3)} m`,
      `${landmark.world_point.z_metres.toFixed(3)} m${landmark.z_source ? ` · ${landmark.z_source}` : ""}`,
      remove,
    ]);
    ui.landmarkTable.append(record);
  });
}

function row(values) {
  const element = document.createElement("div");
  element.className = "landmark-row";
  values.forEach((value) => {
    if (value instanceof HTMLElement) element.append(value);
    else {
      const cell = document.createElement("span");
      cell.textContent = value;
      element.append(cell);
    }
  });
  return element;
}

async function removeLandmark(landmarkId) {
  if (!confirm(`Remove ${landmarkId} from the current revision? Earlier revisions remain in history.`)) return;
  try {
    setState(await api(`/api/landmarks/${encodeURIComponent(landmarkId)}`, { method: "DELETE" }));
    toast("Landmark removed from the current revision.");
  } catch (error) {
    toast(error.message, true);
  }
}

function worldXY(point) {
  const matrix = state.facility_reference.world_from_plan_pixel;
  return {
    x_metres: matrix[0][0] * point.u + matrix[0][1] * point.v + matrix[0][2],
    y_metres: matrix[1][0] * point.u + matrix[1][1] * point.v + matrix[1][2],
  };
}

function approvedFrame() {
  if (!state || !derived.approved_frame_id) return null;
  return state.frames.find((frame) => frame.frame_id === derived.approved_frame_id) || null;
}

function clearDraft() {
  draftImagePoint = null;
  draftPlanPoint = null;
  interaction = null;
}

async function exportSnapshot() {
  try {
    const payload = await api("/api/export", { method: "POST" });
    const blob = new Blob([JSON.stringify(payload.export, null, 2) + "\n"], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = payload.filename;
    link.click();
    URL.revokeObjectURL(link.href);
    toast(`Exported ${payload.filename}`);
  } catch (error) {
    toast(error.message, true);
  }
}

async function exportD034Validation() {
  try {
    const payload = await api("/api/export-d034-validation", { method: "POST" });
    const blob = new Blob([JSON.stringify(payload.validation_seal, null, 2) + "\n"], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = payload.filename;
    link.click();
    URL.revokeObjectURL(link.href);
    toast(`Sealed ${payload.filename}; do not inspect before the solve manifest is frozen.`);
  } catch (error) {
    toast(error.message, true);
  }
}

function svg(tag, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

async function api(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

let toastTimer = null;
function toast(message, error = false) {
  clearTimeout(toastTimer);
  ui.toast.textContent = message;
  ui.toast.className = `toast show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { ui.toast.className = "toast"; }, 3600);
}
