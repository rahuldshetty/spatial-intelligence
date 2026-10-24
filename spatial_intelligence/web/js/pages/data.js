/* Data tab: refresh, local import, URL download, and the workspace file tree. */

import {
  checkImportUrl,
  deleteWorkspaceFile,
  getState,
  importFiles,
  importUrl as postImportUrl,
  renameWorkspaceFile,
} from "../api.js";
import { applySnapshot, state } from "../store.js";
import { closeDialog, el, formatBytes, openDialog, toast } from "../dom.js";
import { icon } from "../icons.js";
import { renderFileTree } from "../components/file-tree.js";
import { offerDroneImport } from "../components/drone-import.js";

const HARNESS_FILES = new Set([
  "notebook.ipynb",
  "workspace.json",
  "maps/current.geolibre.json",
]);

/** Workspace files the user may manage; app state and traces are hidden. */
export function manageableWorkspaceFiles(files = state.files) {
  return (files || []).filter(
    (path) =>
      typeof path === "string" &&
      !HARNESS_FILES.has(path) &&
      path !== "traces" &&
      !path.startsWith("traces/")
  );
}

export function renderDataTab() {
  const wrap = el("div", { class: "data-tab" });

  if (!state.active_workspace) {
    const empty = el("div", { class: "empty-hint" });
    empty.append(
      icon("globe", { size: 26, class: "empty-icon" }),
      el("strong", { class: "empty-title", text: "No workspace open — File → New or Open" }),
      el("span", {
        class: "empty-detail",
        text: "Open a workspace to import imagery, vector data, or any file the agent can read.",
      })
    );
    wrap.append(empty);
    return wrap;
  }

  const files = manageableWorkspaceFiles();
  const heading = el("div", { class: "data-heading" });
  const label = el("div", { class: "data-heading-copy" });
  label.append(
    el("strong", { class: "data-heading-label", text: "Workspace files" }),
    el("span", {
      class: "data-count",
      text: files.length + (files.length === 1 ? " file" : " files"),
    })
  );
  const refresh = el("button", {
    class: "data-refresh",
    type: "button",
    title: "Refresh workspace files",
    "aria-label": "Refresh workspace files",
    onclick: () => refreshWorkspaceData(),
  });
  refresh.append(icon("refresh-cw", { size: 15 }));
  heading.append(label, refresh);
  wrap.append(heading);

  const localRow = el("div", { class: "import-row" });
  const filesInput = el("input", { type: "file", multiple: "multiple", style: "display:none" });
  const folderInput = el("input", { type: "file", webkitdirectory: "", style: "display:none" });
  filesInput.addEventListener("change", () => importLocal(filesInput));
  folderInput.addEventListener("change", () => importLocal(folderInput));
  localRow.append(
    filesInput,
    folderInput,
    importButton("upload", "Import files…", () => filesInput.click()),
    importButton("folder-open", "Import folder…", () => folderInput.click()),
    importButton("download", "Add from URL…", () => openUrlDialog())
  );
  wrap.append(localRow);

  const dropZone = el("div", { class: "drop-zone" });
  dropZone.append(
    icon("upload", { size: 18 }),
    el("span", { text: "Drop files here to import them into " }),
    el("strong", { text: "data/" })
  );
  dropZone.addEventListener("click", () => filesInput.click());
  wrap.append(dropZone);
  attachDrop(wrap);

  wrap.append(
    el("p", {
      class: "data-note",
      text: "The notebook, map snapshot, and run traces are app-managed and hidden here.",
    })
  );

  if (!files.length) {
    const empty = el("div", { class: "empty-hint" });
    empty.append(
      icon("folder-open", { size: 24, class: "empty-icon" }),
      el("strong", { class: "empty-title", text: "No files yet" }),
      el("span", {
        class: "empty-detail",
        text: "Import local files or drop them above, or ask the agent to download imagery.",
      })
    );
    wrap.append(empty);
  } else {
    wrap.append(
      renderFileTree(files, {
        onRename: (path) => openRenameDialog(path),
        onDelete: (path) => confirmDeleteFile(path),
        onContextMenu: (path, event, isDir) => openTreeMenu(path, event, isDir),
      })
    );
  }

  return wrap;
}

function importButton(name, label, onclick) {
  const button = el("button", { type: "button", onclick });
  button.append(icon(name, { size: 14 }), el("span", { text: label }));
  return button;
}

function attachDrop(target) {
  let depth = 0;
  const activate = (event) => {
    event.preventDefault();
    target.classList.add("drop-active");
  };
  target.addEventListener("dragover", activate);
  target.addEventListener("dragenter", (event) => {
    event.preventDefault();
    depth += 1;
    target.classList.add("drop-active");
  });
  target.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) target.classList.remove("drop-active");
  });
  target.addEventListener("drop", async (event) => {
    event.preventDefault();
    depth = 0;
    target.classList.remove("drop-active");
    const files = event.dataTransfer && event.dataTransfer.files;
    await uploadFiles(files);
  });
}

export async function refreshWorkspaceData() {
  try {
    applySnapshot(await getState());
    toast("Workspace files refreshed", "success");
  } catch (e) {
    toast(e.message || String(e), "error");
  }
}

let treeMenu = null;
let treeMenuDismiss = null;

function closeTreeMenu() {
  if (treeMenuDismiss) {
    treeMenuDismiss();
    treeMenuDismiss = null;
  }
  if (treeMenu) {
    treeMenu.remove();
    treeMenu = null;
  }
}

/** Show the file tree's right-click menu for one file or folder. */
function openTreeMenu(path, event, isDir) {
  closeTreeMenu();
  const menu = el("div", { class: "tree-menu", role: "menu" });
  const item = el("button", {
    type: "button",
    class: "tree-menu-item",
    role: "menuitem",
  });
  item.append(
    icon(isDir ? "folder" : "globe", { size: 14 }),
    el("span", { text: isDir ? "Import drone files to GeoLibre" : "Import to GeoLibre" })
  );
  item.addEventListener("click", () => {
    closeTreeMenu();
    offerDroneImport([path], { notifyIfEmpty: true });
  });
  menu.append(item);
  document.body.append(menu);

  const bounds = menu.getBoundingClientRect();
  const left = Math.max(8, Math.min(event.clientX, window.innerWidth - bounds.width - 8));
  const top = Math.max(8, Math.min(event.clientY, window.innerHeight - bounds.height - 8));
  menu.style.left = left + "px";
  menu.style.top = top + "px";

  const dismiss = (pointer) => {
    if (!menu.contains(pointer.target)) closeTreeMenu();
  };
  const onKey = (keyboard) => {
    if (keyboard.key === "Escape") closeTreeMenu();
  };
  document.addEventListener("pointerdown", dismiss, true);
  document.addEventListener("keydown", onKey);
  treeMenuDismiss = () => {
    document.removeEventListener("pointerdown", dismiss, true);
    document.removeEventListener("keydown", onKey);
  };
  treeMenu = menu;
  item.focus();
}

async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) {
    toast("Choose a file or folder first", "warning");
    return;
  }
  try {
    const result = await importFiles(fileList);
    const names = result.imported || [];
    applySnapshot(await getState());
    toast("Imported " + names.length + " file(s)", "success");
    if (names.length) await offerDroneImport(names);
  } catch (e) {
    toast(e.message || String(e), "error");
  }
}

async function importLocal(fileInput) {
  await uploadFiles(fileInput.files);
  fileInput.value = "";
}

export function openRenameDialog(path) {
  const base = path.split("/").pop();
  openDialog((dialog) => {
    const heading = el("div", { class: "dialog-heading" });
    heading.append(
      el("h3", { text: "Rename file" }),
      el("p", {
        class: "dialog-subtitle",
        text: "The file stays in its current folder. Enter a new name.",
      })
    );
    const input = el("input", {
      type: "text",
      value: base,
      autocomplete: "off",
      spellcheck: "false",
      "aria-label": "New file name",
    });
    const status = el("p", { class: "dialog-status", text: path });
    const rename = el("button", { class: "primary", type: "button", text: "Rename" });
    const actions = el("div", { class: "dialog-actions" });
    actions.append(
      el("button", { type: "button", text: "Cancel", onclick: closeDialog }),
      rename
    );

    const submit = async () => {
      const name = input.value.trim();
      if (!name) {
        status.className = "dialog-status error";
        status.textContent = "Enter a file name.";
        return;
      }
      if (name.includes("/") || name.includes("\\")) {
        status.className = "dialog-status error";
        status.textContent = "A file name cannot contain a path separator.";
        return;
      }
      rename.disabled = true;
      input.disabled = true;
      status.className = "dialog-status";
      status.textContent = "Renaming…";
      try {
        applySnapshot(await renameWorkspaceFile(path, name));
        closeDialog();
        toast("Renamed to " + name, "success");
      } catch (error) {
        status.className = "dialog-status error";
        status.textContent = error.message || String(error);
        rename.disabled = false;
        input.disabled = false;
      }
    };

    rename.addEventListener("click", submit);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        submit();
      }
    });
    dialog.append(heading, input, status, actions);
    window.requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
  });
}

export function confirmDeleteFile(path) {
  openDialog((dialog) => {
    const heading = el("div", { class: "dialog-heading" });
    heading.append(
      el("h3", { text: "Delete file" }),
      el("p", {
        class: "dialog-subtitle",
        text: "This removes the file from the workspace.",
      })
    );
    const warning = el("div", { class: "dialog-warning" });
    warning.append(
      el("strong", { text: path }),
      el("span", {
        text:
          "This cannot be undone. If a map layer uses this file, the layer is removed too.",
      })
    );
    const status = el("p", { class: "dialog-status" });
    const remove = el("button", { class: "danger", type: "button", text: "Delete" });
    const actions = el("div", { class: "dialog-actions" });
    actions.append(
      el("button", { type: "button", text: "Cancel", onclick: closeDialog }),
      remove
    );

    remove.addEventListener("click", async () => {
      remove.disabled = true;
      status.className = "dialog-status";
      status.textContent = "Deleting…";
      try {
        applySnapshot(await deleteWorkspaceFile(path));
        closeDialog();
        toast("Deleted " + path.split("/").pop(), "success");
      } catch (error) {
        status.className = "dialog-status error";
        status.textContent = error.message || String(error);
        remove.disabled = false;
      }
    });

    dialog.append(heading, warning, status, actions);
  });
}

export function openUrlDialog() {
  openDialog((dialog) => {
    const heading = el("div", { class: "dialog-heading" });
    heading.append(
      el("h3", { text: "Download from URL" }),
      el("p", {
        class: "dialog-subtitle",
        text: "The server checks the link, then downloads it into the workspace's data/ folder.",
      })
    );
    const input = el("input", {
      type: "url",
      placeholder: "https://example.com/file.tif",
      "aria-label": "URL to download into the workspace",
      autocomplete: "off",
    });
    const status = el("p", { class: "dialog-status" });
    const download = el("button", {
      class: "primary",
      type: "button",
      text: "Download",
    });
    const actions = el("div", { class: "dialog-actions" });
    actions.append(
      el("button", { type: "button", text: "Cancel", onclick: closeDialog }),
      download
    );

    const setStatus = (text, kind = "") => {
      status.textContent = text;
      status.className = "dialog-status" + (kind ? " " + kind : "");
    };

    const run = async () => {
      const url = input.value.trim();
      if (!url) {
        setStatus("Enter a URL first.", "error");
        input.focus();
        return;
      }
      download.disabled = true;
      input.disabled = true;
      setStatus("Checking reachability…");
      try {
        const check = await checkImportUrl(url);
        if (!check.ok) {
          setStatus("Unreachable: " + (check.error || "no response"), "error");
          download.disabled = false;
          input.disabled = false;
          return;
        }
        const size = check.content_length ? " (" + formatBytes(check.content_length) + ")" : "";
        setStatus("Downloading " + (check.filename || "file") + size + "…");
        const imported = await postImportUrl(url, check.filename || null);
        applySnapshot(await getState());
        closeDialog();
        toast("Downloaded into the workspace data folder", "success");
        if (imported.path) await offerDroneImport([imported.path]);
      } catch (error) {
        setStatus(error.message || String(error), "error");
        download.disabled = false;
        input.disabled = false;
      }
    };

    download.addEventListener("click", run);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        run();
      }
    });
    dialog.append(heading, input, status, actions);
    window.requestAnimationFrame(() => input.focus());
  });
}

export function renderDataOnly() {
  const content = document.getElementById("tab-content");
  if (content && state.selected_tab === "Data") {
    content.replaceChildren(renderDataTab());
  }
}
