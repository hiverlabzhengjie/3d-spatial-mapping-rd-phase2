"use strict";

const CAMERA_IDS = ["office-cam-01", "office-cam-02", "office-cam-03", "office-cam-04"];
const CAMERA_COLORS = {
  "office-cam-01": "#e4473e",
  "office-cam-02": "#28a75b",
  "office-cam-03": "#2879d7",
  "office-cam-04": "#e2a522",
};

const ui = {};
let registration = null;
let derived = null;
let selectedCameraId = CAMERA_IDS[0];
let interaction = null;
let drag = null;
let zoom = 1;
let dirty = false;

document.addEventListener("DOMContentLoaded", async () => {
  [
    "uploadView", "workspaceView", "planFile", "replacePlanFile", "uploadMessage", "saveButton",
    "exportButton", "saveStatus", "planFilename", "planHash", "scaleId", "scaleDistance",
    "scaleMeaning", "scaleUncertainty", "scaleSource", "drawScaleButton", "scaleList",
    "scaleSummary", "originMeaning", "placeOriginButton", "setXAxisButton", "resetOriginButton",
    "planViewport", "planScene", "planImage", "planOverlay", "interactionTitle", "interactionHint",
    "zoomOutButton", "zoomInButton", "zoomFitButton", "zoomValue", "cursorReadout", "cameraTabs",
    "selectedCameraDot", "selectedCameraTitle", "cameraStatus", "cameraLabel", "cameraReference",
    "cameraHeight", "cameraHeightUncertainty", "placeCameraButton", "setPanButton",
    "cameraCoordinates", "endpointStatus", "rtspUrl", "saveEndpointButton", "toast",
  ].forEach((id) => { ui[id] = document.getElementById(id); });
  wireEvents();
  await loadInitialState();
});

function wireEvents() {
  ui.planFile.addEventListener("change", (event) => uploadSelectedFile(event.target.files[0]));
  ui.replacePlanFile.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file && confirm("Replace the plan? Current state remains in revision history, but a clean registration will start.")) {
      await uploadSelectedFile(file);
    }
  });
  ui.saveButton.addEventListener("click", saveRegistration);
  ui.exportButton.addEventListener("click", exportSnapshot);
  ui.drawScaleButton.addEventListener("click", beginScaleControl);
  ui.placeOriginButton.addEventListener("click", () => setInteraction("origin", "Place origin", "Click the exact permanent origin feature."));
  ui.resetOriginButton.addEventListener("click", () => {
    if (!confirm("Repositioning the origin will recalculate all displayed camera XY coordinates. Saved revisions remain recoverable. Continue?")) return;
    setInteraction("origin", "Reposition origin", "Click the new permanent origin feature.");
  });
  ui.setXAxisButton.addEventListener("click", () => {
    if (!registration.frame) return toast("Place the origin first.", true);
    setInteraction("x-axis", "Choose +X", "Click in the desired positive-X direction from the origin.");
  });
  ui.placeCameraButton.addEventListener("click", () => {
    updateSelectedCameraForm();
    setInteraction("camera", `Place ${shortCamera(selectedCameraId)}`, "Click the physical mounting reference, then drag the dot to refine.");
  });
  ui.setPanButton.addEventListener("click", () => {
    updateSelectedCameraForm();
    if (!cameraById(selectedCameraId).marker) return toast("Place this camera first.", true);
    setInteraction("pan", `Rough pan for ${shortCamera(selectedCameraId)}`, "Click the approximate viewing direction. This remains display-only.");
  });
  ui.saveEndpointButton.addEventListener("click", saveEndpoint);
  ui.zoomInButton.addEventListener("click", () => setZoom(zoom + .15));
  ui.zoomOutButton.addEventListener("click", () => setZoom(zoom - .15));
  ui.zoomFitButton.addEventListener("click", fitPlan);
  ui.planOverlay.addEventListener("pointerdown", overlayPointerDown);
  ui.planOverlay.addEventListener("pointermove", overlayPointerMove);
  ui.planOverlay.addEventListener("pointerup", overlayPointerUp);
  ui.planOverlay.addEventListener("pointercancel", overlayPointerUp);
  [ui.cameraLabel, ui.cameraReference, ui.cameraHeight, ui.cameraHeightUncertainty].forEach((element) => {
    element.addEventListener("change", updateSelectedCameraForm);
  });
  ui.originMeaning.addEventListener("change", () => {
    if (!registration.frame) return;
    registration.frame.origin_feature_meaning = ui.originMeaning.value.trim();
    markDirty();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!dirty) return;
    event.preventDefault();
  });
}

async function loadInitialState() {
  try {
    const status = await api("/api/status");
    if (!status.has_state) return showUpload();
    setState(status.state);
  } catch (error) {
    showUpload();
    ui.uploadMessage.textContent = error.message;
  }
}

async function uploadSelectedFile(file) {
  if (!file) return;
  ui.uploadMessage.textContent = "Rendering page 1 locally…";
  try {
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "X-Filename": file.name, "Content-Type": "application/pdf" },
      body: file,
    });
    const payload = await readResponse(response);
    setState(payload.state);
    toast("Plan uploaded and hash-bound.");
  } catch (error) {
    ui.uploadMessage.textContent = error.message;
    toast(error.message, true);
  } finally {
    ui.planFile.value = "";
    ui.replacePlanFile.value = "";
  }
}

function showUpload() {
  ui.uploadView.hidden = false;
  ui.workspaceView.hidden = true;
}

function setState(payload) {
  registration = payload;
  derived = payload.derived || computeDerived();
  delete registration.derived;
  dirty = false;
  interaction = null;
  ui.uploadView.hidden = true;
  ui.workspaceView.hidden = false;
  ui.saveButton.disabled = false;
  ui.exportButton.disabled = false;
  ui.planFilename.textContent = registration.plan.original_filename;
  ui.planHash.textContent = registration.plan.source_sha256;
  ui.planImage.src = `/api/plan-image?v=${registration.plan.source_sha256}`;
  ui.planScene.style.width = `${registration.plan.image_width_pixels}px`;
  ui.planScene.style.height = `${registration.plan.image_height_pixels}px`;
  ui.planOverlay.setAttribute("viewBox", `0 0 ${registration.plan.image_width_pixels} ${registration.plan.image_height_pixels}`);
  ui.planOverlay.setAttribute("width", registration.plan.image_width_pixels);
  ui.planOverlay.setAttribute("height", registration.plan.image_height_pixels);
  ui.planImage.onload = fitPlan;
  renderAll();
  loadSelectedEndpoint();
  ui.saveStatus.textContent = `Revision ${registration.revision} saved`;
}

function renderAll() {
  derived = computeDerived();
  renderScaleControls();
  renderCameraTabs();
  renderCameraForm();
  renderOverlay();
  renderScaleSummary();
}

function renderScaleControls() {
  ui.scaleList.replaceChildren();
  registration.scale_controls.forEach((control) => {
    const row = document.createElement("div");
    row.className = "compact-item";
    const description = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = control.control_id;
    const detail = document.createElement("span");
    detail.textContent = `${control.distance_metres.toFixed(3)} m · ${control.source_kind}`;
    description.append(strong, detail);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      registration.scale_controls = registration.scale_controls.filter((item) => item.control_id !== control.control_id);
      markDirty();
      renderAll();
    });
    row.append(description, remove);
    ui.scaleList.append(row);
  });
}

function renderScaleSummary() {
  const ppm = derived.pixels_per_metre;
  if (!ppm) {
    ui.scaleSummary.className = "metric-box muted";
    ui.scaleSummary.textContent = "No metric scale yet";
    return;
  }
  const spread = derived.scale_spread_fraction;
  ui.scaleSummary.className = "metric-box";
  ui.scaleSummary.textContent = `${ppm.toFixed(2)} px/m · ${registration.scale_controls.length} control${registration.scale_controls.length === 1 ? "" : "s"}${spread == null ? " · provisional single check" : ` · ${(spread * 100).toFixed(2)}% scale spread`}`;
}

function renderCameraTabs() {
  ui.cameraTabs.replaceChildren();
  CAMERA_IDS.forEach((cameraId, index) => {
    const button = document.createElement("button");
    button.className = `camera-tab${cameraId === selectedCameraId ? " active" : ""}`;
    button.innerHTML = `<i style="background:${CAMERA_COLORS[cameraId]}"></i>C${index + 1}`;
    button.addEventListener("click", () => {
      updateSelectedCameraForm();
      selectedCameraId = cameraId;
      interaction = null;
      renderCameraTabs();
      renderCameraForm();
      loadSelectedEndpoint();
      updateInteractionHeader();
    });
    ui.cameraTabs.append(button);
  });
}

function renderCameraForm() {
  const camera = cameraById(selectedCameraId);
  const cameraDerived = derived.cameras[selectedCameraId];
  ui.selectedCameraDot.style.background = CAMERA_COLORS[selectedCameraId];
  ui.selectedCameraTitle.textContent = shortCamera(selectedCameraId);
  ui.cameraStatus.textContent = cameraDerived.status.replaceAll("-", " ");
  ui.cameraLabel.value = camera.physical_label || "";
  ui.cameraReference.value = camera.reference_meaning || "";
  ui.cameraHeight.value = camera.mounting_height_metres ?? "";
  ui.cameraHeightUncertainty.value = camera.height_uncertainty_metres ?? "0.10";
  if (cameraDerived.world_xy) {
    const z = camera.mounting_height_metres == null ? "Z —" : `Z ${camera.mounting_height_metres.toFixed(2)} m`;
    ui.cameraCoordinates.textContent = `X ${cameraDerived.world_xy.x_metres.toFixed(3)} m · Y ${cameraDerived.world_xy.y_metres.toFixed(3)} m · ${z}`;
  } else {
    ui.cameraCoordinates.textContent = camera.marker ? "Place scale and origin to calculate XY" : "Camera not placed";
  }
  renderEndpointState(cameraDerived);
}

function updateSelectedCameraForm() {
  const camera = cameraById(selectedCameraId);
  const physicalLabel = ui.cameraLabel.value.trim();
  const referenceMeaning = ui.cameraReference.value.trim();
  const mountingHeight = optionalNumber(ui.cameraHeight.value);
  const heightUncertainty = mountingHeight == null ? null : optionalNumber(ui.cameraHeightUncertainty.value);
  const changed = camera.physical_label !== physicalLabel ||
    camera.reference_meaning !== referenceMeaning ||
    camera.mounting_height_metres !== mountingHeight ||
    camera.height_uncertainty_metres !== heightUncertainty;
  if (!changed) return;
  camera.physical_label = physicalLabel;
  camera.reference_meaning = referenceMeaning;
  camera.mounting_height_metres = mountingHeight;
  camera.height_uncertainty_metres = heightUncertainty;
  markDirty();
  derived = computeDerived();
}

function renderEndpointState(cameraDerived) {
  ui.endpointStatus.textContent = cameraDerived.endpoint_configured ? "Configured locally" : "Not configured";
  ui.endpointStatus.style.color = cameraDerived.endpoint_configured ? "#176b52" : "#a46522";
  ui.cameraStatus.textContent = cameraDerived.status.replaceAll("-", " ");
}

function beginScaleControl() {
  const controlId = ui.scaleId.value.trim();
  const distance = optionalNumber(ui.scaleDistance.value);
  const uncertainty = optionalNumber(ui.scaleUncertainty.value);
  if (!/^[a-z][a-z0-9-]*$/.test(controlId)) return toast("Control ID must be lowercase and hyphenated.", true);
  if (registration.scale_controls.some((control) => control.control_id === controlId)) return toast("Scale control ID already exists.", true);
  if (!(distance > 0) || uncertainty == null || uncertainty < 0) return toast("Enter a positive distance and non-negative uncertainty.", true);
  const meaning = ui.scaleMeaning.value.trim();
  if (!meaning) return toast("Describe the scale endpoints.", true);
  interaction = {
    type: "scale",
    points: [],
    draft: { control_id: controlId, meaning, distance_metres: distance, distance_uncertainty_metres: uncertainty, source_kind: ui.scaleSource.value },
  };
  updateInteractionHeader("Draw scale control", "Click endpoint A, then endpoint B.");
  setActiveTool(ui.drawScaleButton);
}

function setInteraction(type, title, hint) {
  interaction = { type };
  updateInteractionHeader(title, hint);
  const button = type === "origin" ? ui.placeOriginButton : type === "x-axis" ? ui.setXAxisButton : type === "camera" ? ui.placeCameraButton : ui.setPanButton;
  setActiveTool(button);
}

function setActiveTool(active) {
  [ui.drawScaleButton, ui.placeOriginButton, ui.setXAxisButton, ui.placeCameraButton, ui.setPanButton].forEach((button) => button.classList.toggle("active", button === active));
}

function updateInteractionHeader(title = "Inspect plan", hint = "Choose a tool, then click or drag on the plan.") {
  ui.interactionTitle.textContent = title;
  ui.interactionHint.textContent = hint;
  if (!interaction) setActiveTool(null);
}

function overlayPointerDown(event) {
  const target = event.target.closest("[data-drag-role]");
  if (!target) return;
  event.preventDefault();
  const role = target.dataset.dragRole;
  const cameraId = target.dataset.cameraId || null;
  drag = { role, cameraId, moved: false };
  ui.planOverlay.setPointerCapture(event.pointerId);
}

function overlayPointerMove(event) {
  const point = pointFromEvent(event);
  updateCursor(point);
  if (!drag) {
    if (interaction?.type === "camera") showCameraPlacementPreview(point);
    return;
  }
  drag.moved = true;
  if (drag.role === "origin" && registration.frame) registration.frame.origin = point;
  if (drag.role === "x-axis" && registration.frame) registration.frame.positive_x_handle = point;
  if (drag.role === "camera" && drag.cameraId) cameraById(drag.cameraId).marker = point;
  if (drag.role === "pan" && drag.cameraId) cameraById(drag.cameraId).rough_pan_endpoint = point;
  markDirty();
  renderOverlay();
  derived = computeDerived();
  renderCameraForm();
}

function showCameraPlacementPreview(point) {
  const world = worldXY(point);
  if (!world) {
    ui.cameraCoordinates.textContent = "Preview unavailable: place scale and origin first";
    return;
  }
  const camera = cameraById(selectedCameraId);
  const z = camera.mounting_height_metres == null ? "Z —" : `Z ${camera.mounting_height_metres.toFixed(2)} m`;
  ui.cameraCoordinates.textContent = `Preview X ${world.x_metres.toFixed(3)} m · Y ${world.y_metres.toFixed(3)} m · ${z} · click to place`;
}

function overlayPointerUp(event) {
  const point = pointFromEvent(event);
  if (drag) {
    drag = null;
    if (ui.planOverlay.hasPointerCapture(event.pointerId)) ui.planOverlay.releasePointerCapture(event.pointerId);
    renderAll();
    return;
  }
  if (!interaction) return;
  if (interaction.type === "scale") {
    interaction.points.push(point);
    if (interaction.points.length === 2) {
      const [pointA, pointB] = interaction.points;
      registration.scale_controls.push({ ...interaction.draft, point_a: pointA, point_b: pointB });
      markDirty();
      interaction = null;
      updateInteractionHeader();
      renderAll();
    } else {
      updateInteractionHeader("Draw scale control", "Endpoint A set. Click endpoint B.");
      renderOverlay();
    }
    return;
  }
  if (interaction.type === "origin") {
    const handleU = Math.max(0, point.u - Math.min(140, registration.plan.image_width_pixels * .08));
    registration.frame = { origin: point, positive_x_handle: { u: handleU, v: point.v }, origin_feature_meaning: ui.originMeaning.value.trim() || "user-selected permanent origin feature" };
  } else if (interaction.type === "x-axis") {
    registration.frame.positive_x_handle = point;
  } else if (interaction.type === "camera") {
    cameraById(selectedCameraId).marker = point;
  } else if (interaction.type === "pan") {
    cameraById(selectedCameraId).rough_pan_endpoint = point;
  }
  markDirty();
  interaction = null;
  updateInteractionHeader();
  renderAll();
}

function renderOverlay() {
  ui.planOverlay.replaceChildren();
  const defs = svg("defs");
  CAMERA_IDS.forEach((cameraId) => {
    const marker = svg("marker", { id: `arrow-${cameraId}`, markerWidth: 10, markerHeight: 10, refX: 8, refY: 3, orient: "auto", markerUnits: "strokeWidth" });
    marker.append(svg("path", { d: "M0,0 L0,6 L9,3 z", fill: CAMERA_COLORS[cameraId] }));
    defs.append(marker);
  });
  ui.planOverlay.append(defs);
  registration.scale_controls.forEach((control) => {
    ui.planOverlay.append(svg("line", { x1: control.point_a.u, y1: control.point_a.v, x2: control.point_b.u, y2: control.point_b.v, class: "overlay-scale" }));
    ui.planOverlay.append(textSvg((control.point_a.u + control.point_b.u) / 2, (control.point_a.v + control.point_b.v) / 2 - 10, `${control.control_id} · ${control.distance_metres.toFixed(3)} m`));
  });
  if (interaction?.type === "scale" && interaction.points.length === 1) {
    ui.planOverlay.append(svg("circle", { cx: interaction.points[0].u, cy: interaction.points[0].v, r: 9, fill: "#176b52" }));
  }
  if (registration.frame) {
    const { origin, positive_x_handle: xh } = registration.frame;
    const dx = xh.u - origin.u;
    const dy = xh.v - origin.v;
    const length = Math.hypot(dx, dy);
    const axisLength = Math.max(length, 100);
    const xu = dx / length;
    const xv = dy / length;
    const yu = xv;
    const yv = -xu;
    const yEnd = { u: origin.u + yu * axisLength, v: origin.v + yv * axisLength };
    ui.planOverlay.append(svg("line", { x1: origin.u, y1: origin.v, x2: xh.u, y2: xh.v, class: "overlay-axis-x" }));
    ui.planOverlay.append(svg("line", { x1: origin.u, y1: origin.v, x2: yEnd.u, y2: yEnd.v, class: "overlay-axis-y" }));
    ui.planOverlay.append(textSvg(xh.u + 10, xh.v, "+X"));
    ui.planOverlay.append(textSvg(yEnd.u + 10, yEnd.v, "+Y"));
    ui.planOverlay.append(svg("circle", { cx: origin.u, cy: origin.v, r: 13, class: "overlay-origin", "data-drag-role": "origin" }));
    ui.planOverlay.append(svg("circle", { cx: xh.u, cy: xh.v, r: 10, class: "overlay-handle", "data-drag-role": "x-axis" }));
  }
  registration.cameras.forEach((camera, index) => {
    if (!camera.marker) return;
    const color = CAMERA_COLORS[camera.camera_id];
    if (camera.rough_pan_endpoint) {
      ui.planOverlay.append(svg("line", { x1: camera.marker.u, y1: camera.marker.v, x2: camera.rough_pan_endpoint.u, y2: camera.rough_pan_endpoint.v, class: "overlay-pan", stroke: color, "marker-end": `url(#arrow-${camera.camera_id})`, "data-drag-role": "pan", "data-camera-id": camera.camera_id }));
    }
    ui.planOverlay.append(svg("circle", { cx: camera.marker.u, cy: camera.marker.v, r: 14, fill: color, class: "overlay-camera", "data-drag-role": "camera", "data-camera-id": camera.camera_id }));
    ui.planOverlay.append(textSvg(camera.marker.u + 18, camera.marker.v - 16, `C${index + 1}`));
  });
}

function pointFromEvent(event) {
  const svgPoint = ui.planOverlay.createSVGPoint();
  svgPoint.x = event.clientX;
  svgPoint.y = event.clientY;
  const local = svgPoint.matrixTransform(ui.planOverlay.getScreenCTM().inverse());
  return {
    u: clamp(local.x, 0, registration.plan.image_width_pixels),
    v: clamp(local.y, 0, registration.plan.image_height_pixels),
  };
}

function updateCursor(point) {
  let text = `Pixel ${point.u.toFixed(1)}, ${point.v.toFixed(1)}`;
  const world = worldXY(point);
  if (world) text += ` · World X ${world.x_metres.toFixed(3)} m, Y ${world.y_metres.toFixed(3)} m`;
  ui.cursorReadout.textContent = text;
}

function computeDerived() {
  if (!registration) return null;
  const ratios = registration.scale_controls.map((control) => pixelDistance(control.point_a, control.point_b) / control.distance_metres);
  const ppm = ratios.length ? ratios.reduce((sum, value) => sum + value, 0) / ratios.length : null;
  const spread = ratios.length < 2 ? null : (Math.max(...ratios) - Math.min(...ratios)) / ppm;
  const cameras = {};
  registration.cameras.forEach((camera) => {
    const endpointConfigured = derived?.cameras?.[camera.camera_id]?.endpoint_configured || false;
    let status = "unplaced";
    if (camera.marker) status = "placed";
    if (camera.marker && camera.physical_label && camera.mounting_height_metres != null) status = endpointConfigured ? "ready-for-calibration" : "mount-prior-complete";
    cameras[camera.camera_id] = { status, world_xy: camera.marker ? worldXY(camera.marker, ppm) : null, endpoint_configured: endpointConfigured };
  });
  return { pixels_per_metre: ppm, scale_spread_fraction: spread, frame_ready: Boolean(ppm && registration.frame), cameras };
}

function worldXY(point, explicitPpm = null) {
  const ppm = explicitPpm || derived?.pixels_per_metre;
  if (!ppm || !registration.frame) return null;
  const origin = registration.frame.origin;
  const handle = registration.frame.positive_x_handle;
  const dx = handle.u - origin.u;
  const dy = handle.v - origin.v;
  const length = Math.hypot(dx, dy);
  if (length <= 10) return null;
  const xu = dx / length;
  const xv = dy / length;
  const yu = xv;
  const yv = -xu;
  const du = point.u - origin.u;
  const dv = point.v - origin.v;
  return { x_metres: (du * xu + dv * xv) / ppm, y_metres: (du * yu + dv * yv) / ppm };
}

async function saveRegistration() {
  updateSelectedCameraForm();
  try {
    const payload = await api("/api/state", { method: "PUT", body: JSON.stringify(registration) });
    setState(payload);
    toast(`Revision ${registration.revision} saved.`);
    return true;
  } catch (error) {
    toast(error.message, true);
    return false;
  }
}

async function saveEndpoint() {
  const rtspUrl = ui.rtspUrl.value;
  if (!rtspUrl) return toast("Enter an RTSP URL or leave it for later.", true);
  const cameraId = selectedCameraId;
  try {
    await api(`/api/cameras/${cameraId}/endpoint`, { method: "PUT", body: JSON.stringify({ rtsp_url: rtspUrl }) });
    derived.cameras[cameraId].endpoint_configured = true;
    derived = computeDerived();
    if (selectedCameraId === cameraId) renderEndpointState(derived.cameras[cameraId]);
    toast(`${shortCamera(cameraId)} endpoint stored locally; registration work was preserved.`);
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadSelectedEndpoint() {
  const cameraId = selectedCameraId;
  ui.rtspUrl.value = "";
  try {
    const payload = await api(`/api/cameras/${cameraId}/endpoint`);
    if (selectedCameraId !== cameraId) return;
    ui.rtspUrl.value = payload.rtsp_url;
    derived.cameras[cameraId].endpoint_configured = payload.configured;
    derived = computeDerived();
    renderEndpointState(derived.cameras[cameraId]);
  } catch (error) {
    if (selectedCameraId === cameraId) toast(error.message, true);
  }
}

async function exportSnapshot() {
  try {
    if (dirty && !(await saveRegistration())) return;
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

function markDirty() {
  dirty = true;
  ui.saveStatus.textContent = `Revision ${registration.revision} · unsaved changes`;
}

function setZoom(value) {
  zoom = clamp(value, .25, 2.5);
  ui.planScene.style.transform = `scale(${zoom})`;
  ui.planScene.style.marginBottom = `${registration.plan.image_height_pixels * (zoom - 1)}px`;
  ui.planScene.style.marginRight = `${registration.plan.image_width_pixels * (zoom - 1)}px`;
  ui.zoomValue.textContent = `${Math.round(zoom * 100)}%`;
}

function fitPlan() {
  if (!registration) return;
  const availableWidth = ui.planViewport.clientWidth - 48;
  const availableHeight = ui.planViewport.clientHeight - 48;
  setZoom(Math.min(availableWidth / registration.plan.image_width_pixels, availableHeight / registration.plan.image_height_pixels, 1));
  ui.planViewport.scrollTo(0, 0);
}

function cameraById(cameraId) {
  return registration.cameras.find((camera) => camera.camera_id === cameraId);
}

function shortCamera(cameraId) {
  return `Camera ${Number(cameraId.slice(-2))}`;
}

function pixelDistance(a, b) {
  return Math.hypot(b.u - a.u, b.v - a.v);
}

function optionalNumber(value) {
  if (value === "" || value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function svg(tag, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
}

function textSvg(x, y, text) {
  const element = svg("text", { x, y, class: "overlay-label" });
  element.textContent = text;
  return element;
}

async function api(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  return readResponse(response);
}

async function readResponse(response) {
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
