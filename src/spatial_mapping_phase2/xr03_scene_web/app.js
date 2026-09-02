"use strict";

let scenes = [];
let resource = { active: null, queue: [] };
const grid = document.getElementById("scene-grid");
const createDialog = document.getElementById("create-dialog");
const manageDialog = document.getElementById("manage-dialog");

function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]); }
function formatBytes(value) { if (!Number.isFinite(value)) return "No managed files"; if(value<1024)return `${value} B`;if(value<1048576)return `${(value/1024).toFixed(1)} KB`;return `${(value/1048576).toFixed(1)} MB`; }
async function api(url, options={}) { const response=await fetch(url,{cache:"no-store",...options});const payload=await response.json().catch(()=>({}));if(!response.ok)throw new Error(payload.detail||`Request failed (${response.status})`);return payload; }
function notify(message,error=false){const node=document.getElementById("notice");node.textContent=message;node.classList.toggle("error",error);node.classList.remove("hidden");}

function render() {
  const resourceNode=document.getElementById("resource-summary");const active=resource.active;const queued=Array.isArray(resource.queue)?resource.queue:[];resourceNode.classList.toggle("hidden",!active&&!queued.length);resourceNode.textContent=active?`Heavy processing is running in one scene${queued.length?`; ${queued.length} ${queued.length===1?"action is":"actions are"} waiting`:""}. You can keep editing other scenes.`:`${queued.length} processing ${queued.length===1?"action is":"actions are"} waiting for this computer.`;
  if (!scenes.length) { grid.innerHTML='<div class="empty"><h3>No scenes here yet</h3><p>Create one to begin a separate environment.</p></div>'; return; }
  grid.innerHTML=scenes.map(scene=>`<article class="scene-card ${scene.lifecycle}">
    <div class="scene-meta"><span class="badge ${scene.readiness}">${scene.readiness === "ready" ? "Ready" : "Setup in progress"}</span>${scene.lifecycle === "archived" ? '<span class="badge archived">Archived</span>' : ""}${scene.lifecycle === "deletion_failed" ? '<span class="badge failed">Removal needs attention</span>' : ""}</div>
    <h3>${escapeHtml(scene.display_name)}</h3><p>${scene.camera_count} ${scene.camera_count===1?"camera":"cameras"} · ${scene.storage_ownership === "managed" ? "Managed locally" : "Existing storage protected"}</p>
    <div class="scene-actions">${scene.lifecycle === "active" ? `<a href="/scenes/${encodeURIComponent(scene.scene_uuid)}/"><button class="open">Open scene</button></a>` : ""}<button class="secondary manage" data-scene="${escapeHtml(scene.scene_uuid)}">Manage</button></div>
  </article>`).join("");
  document.querySelectorAll(".manage").forEach(button=>button.addEventListener("click",()=>openManage(button.dataset.scene)));
}

async function refresh(){const payload=await api(`/api/scenes?include_archived=${document.getElementById("show-archived").checked}`);scenes=payload.scenes;resource=payload.resource||resource;render();}
function renderCameraNames(){const count=Math.max(1,Math.min(64,Number(document.getElementById("camera-count").value)||1));const node=document.getElementById("camera-names");const old=[...node.querySelectorAll("input")].map(input=>input.value);node.innerHTML=Array.from({length:count},(_,i)=>`<label>Camera ${i+1}<input class="camera-name" maxlength="80" required value="${escapeHtml(old[i]||`Camera ${i+1}`)}"></label>`).join("");}

async function openManage(sceneUuid){
  const scene=scenes.find(item=>item.scene_uuid===sceneUuid);if(!scene)return;
  const content=document.getElementById("manage-content");
  const failed=scene.lifecycle==="deletion_failed";
  const normalControls=`<div class="manage-row"><input id="rename-value" maxlength="80" value="${escapeHtml(scene.display_name)}"><button id="rename-scene" class="secondary">Rename</button></div><button id="archive-scene" class="secondary">${scene.lifecycle==="archived"?"Restore scene":"Archive scene"}</button>`;
  content.innerHTML=`<button type="button" class="close" data-close="manage-dialog">×</button><p class="eyebrow">Manage scene</p><h2>${escapeHtml(scene.display_name)}</h2>${failed?'<div class="danger-note">The last filesystem removal did not finish. The scene and a truthful failure record were retained. Review the exact folder and retry when it is available.</div>':normalControls}<button id="review-delete" class="danger">${failed?"Review and retry removal…":"Remove scene…"}</button><details><summary>Technical storage details</summary><p class="storage">${escapeHtml(scene.workspace_root)}</p></details>`;
  content.querySelector("[data-close]").addEventListener("click",()=>manageDialog.close());
  if(!failed){document.getElementById("rename-scene").addEventListener("click",async()=>{try{await api(`/api/scenes/${scene.scene_uuid}`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({display_name:document.getElementById("rename-value").value,expected_revision:scene.revision})});manageDialog.close();notify("Scene renamed.");await refresh();}catch(error){notify(error.message,true);}});document.getElementById("archive-scene").addEventListener("click",async()=>{try{await api(`/api/scenes/${scene.scene_uuid}`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({archived:scene.lifecycle!=="archived",expected_revision:scene.revision})});manageDialog.close();notify(scene.lifecycle==="archived"?"Scene restored.":"Scene archived. Its files remain unchanged.");await refresh();}catch(error){notify(error.message,true);}});}
  document.getElementById("review-delete").addEventListener("click",()=>reviewDelete(scene));
  manageDialog.showModal();
}

async function reviewDelete(scene){
  try{
    const impact=await api(`/api/scenes/${scene.scene_uuid}/delete-impact`);const content=document.getElementById("manage-content");const storage=impact.storage;
    content.innerHTML=`<button type="button" class="close" data-close="manage-dialog">×</button><p class="eyebrow">Final check</p><h2>Remove ${escapeHtml(scene.display_name)}?</h2><div class="danger-note">${impact.can_delete_files?`This can permanently remove exactly ${storage.file_count} files (${formatBytes(storage.byte_count)}) inside the displayed managed scene folder.`:"This pre-existing workspace is protected. Removing the scene only takes it out of this console; its files stay untouched."}</div>${impact.protected_reasons.map(reason=>`<p>${escapeHtml(reason)}</p>`).join("")}<p class="storage">${escapeHtml(impact.exact_managed_root||scene.workspace_root)}</p><button id="confirm-delete" class="danger">${impact.can_delete_files?"Permanently delete scene and files":"Remove from scene list"}</button><button id="cancel-delete" class="secondary">Cancel</button>`;
    content.querySelector("[data-close]").addEventListener("click",()=>manageDialog.close());document.getElementById("cancel-delete").addEventListener("click",()=>manageDialog.close());document.getElementById("confirm-delete").addEventListener("click",async()=>{if(impact.can_delete_files&&!window.confirm("Permanently delete the exact managed scene folder? This cannot be undone."))return;try{await api(`/api/scenes/${scene.scene_uuid}/delete`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({deletion_token:impact.deletion_token,delete_files:impact.can_delete_files,expected_revision:scene.revision})});manageDialog.close();notify(impact.can_delete_files?"Scene files deleted. A small tombstone was retained.":"Scene removed. Its existing files were not changed.");await refresh();}catch(error){notify(error.message,true);}});
  }catch(error){notify(error.message,true);}
}

document.getElementById("new-scene").addEventListener("click",()=>{document.getElementById("create-form").reset();document.getElementById("camera-count").value="4";renderCameraNames();createDialog.showModal();});
document.getElementById("camera-count").addEventListener("input",renderCameraNames);
document.querySelectorAll("[data-close]").forEach(button=>button.addEventListener("click",()=>document.getElementById(button.dataset.close).close()));
document.getElementById("show-archived").addEventListener("change",()=>refresh().catch(error=>notify(error.message,true)));
document.getElementById("create-form").addEventListener("submit",async event=>{event.preventDefault();try{const payload=await api("/api/scenes",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({display_name:document.getElementById("scene-name").value,camera_names:[...document.querySelectorAll(".camera-name")].map(input=>input.value)})});window.location.assign(payload.open_url);}catch(error){notify(error.message,true);}});
document.getElementById("register-form").addEventListener("submit",async event=>{event.preventDefault();try{const payload=await api("/api/scenes/register",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({workspace_root:document.getElementById("existing-path").value})});notify("Existing scene registered without moving its files.");await refresh();document.getElementById("existing-path").value="";}catch(error){notify(error.message,true);}});
renderCameraNames();refresh().catch(error=>notify(error.message,true));
