/* Menubar: workspace/file menu, cell menu, workspace chip, danger toggle. */

"use strict";

import {
  closeWorkspace,
  openWorkspace,
  runAll,
  saveWorkspace,
  updateSettings,
} from "../api.js";
import { closeDialog, dialogActions, el, openDialog, toast } from "../dom.js";
import { addCell } from "../pages/cells.js";
import { applySnapshot, setState, state } from "../store.js";
import { openSettingsDialog } from "./settings-dialog.js";
import { openWorkflowDialog } from "./workflow-dialog.js";

const ICON_SAFE =
  '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
  '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>';

const ICON_DANGER =
  '<svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor" stroke="none">' +
  '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>';

let openMenu = null;

export function renderMenubar() {
  const bar = el("div", { id: "menubar" });
  bar.append(
    el("div", { class: "brand" }, [
      el("span", { class: "brand-mark", text: "◈" }),
      el("span", { class: "brand-name", text: "Spatial Intelligence" }),
    ])
  );

  const fileMenu = el("div", { class: "menu" });
  const fileBtn = el("button", { text: "File", onclick: () => toggleMenu(fileMenu) });
  const fileItems = el("div", { class: "menu-items" });
  fileItems.append(
    menuItem("New…", () => openNewDialog()),
    menuItem("Open…", () => openOpenDialog()),
    menuItem("Save", () => doSave()),
    menuItem("Close", () => doClose()),
    el("div", { class: "separator" }),
    menuItem("Settings…", () => openSettingsDialog()),
    el("div", { class: "separator" }),
    menuItem("Exit", () => doExit())
  );
  fileMenu.append(fileBtn, fileItems);

  const cellMenu = el("div", { class: "menu" });
  const cellBtn = el("button", { text: "Cell", onclick: () => toggleMenu(cellMenu) });
  const cellItems = el("div", { class: "menu-items" });
  cellItems.append(
    menuItem("Add Markdown Cell", () => runAction(() => addCell("markdown"))),
    menuItem("Add Python Cell", () => runAction(() => addCell("python"))),
    menuItem("Add Prompt Cell", () => runAction(() => addCell("prompt"))),
    el("div", { class: "separator" }),
    menuItem("Run All", () => runAction(() => runAll()))
  );
  cellMenu.append(cellBtn, cellItems);

  bar.append(fileMenu, cellMenu);

  const right = el("div", { class: "menubar-right" });
  if (state.active_workspace) {
    right.append(el("div", { id: "ws-chip", text: state.active_workspace }));
  }
  right.append(dangerToggle());
  bar.append(right);
  return bar;
}

export function closeAllMenus() {
  document.querySelectorAll(".menu.open").forEach((menu) => menu.classList.remove("open"));
  openMenu = null;
}

function toggleMenu(menu) {
  const wasOpen = openMenu === menu;
  closeAllMenus();
  if (!wasOpen) {
    menu.classList.add("open");
    openMenu = menu;
  }
}

function menuItem(label, onclick) {
  return el("button", { text: label, onclick });
}

function dangerToggle() {
  const on = !!(state.settings && state.settings.dangerous_mode);
  const button = el("button", {
    id: "danger-toggle",
    class: "danger-toggle" + (on ? " on" : ""),
    title: on
      ? "Dangerous mode ON — run_python may execute arbitrary commands"
      : "Dangerous mode OFF — run_python is sandboxed",
    "aria-pressed": String(on),
    onclick: () => toggleDangerous(!on),
  });
  button.innerHTML = on ? ICON_DANGER : ICON_SAFE;
  return button;
}

async function toggleDangerous(next) {
  try {
    const settings = await updateSettings({ dangerous_mode: next });
    setState({ settings });
    toast(next ? "Dangerous mode enabled" : "Dangerous mode disabled");
  } catch (error) {
    toast(error.message || String(error));
  }
}

function openNewDialog() {
  openWorkflowDialog();
}

function openOpenDialog() {
  openDialog((dialog) => {
    dialog.append(el("h3", { text: "Open workspace" }));
    const list = el("ul", { id: "ws-list" });
    if (!state.workspaces.length) {
      list.append(el("li", { text: "(no workspaces yet)" }));
    }
    for (const name of state.workspaces) {
      list.append(
        el("li", {
          text: name,
          onclick: async () => {
            closeDialog();
            await runAction(() => openWorkspace(name));
          },
        })
      );
    }
    dialog.append(list);
    dialogActions(dialog, null, null);
  });
}

async function doSave() {
  if (!state.active_workspace) {
    toast("No workspace open");
    return;
  }
  try {
    await saveWorkspace();
    toast("Saved");
  } catch (error) {
    toast(error.message || String(error));
  }
}

function doClose() {
  return runAction(() => closeWorkspace());
}

async function doExit() {
  try {
    await closeWorkspace();
  } catch (_) {
    /* the server stays up even when nothing is open */
  }
  applySnapshot({
    active_workspace: null,
    cells: [],
    files: [],
    jobs: [],
    map_project: null,
    map_app_url: null,
  });
}

async function runAction(action) {
  try {
    const result = await action();
    if (result && typeof result === "object" && "active_workspace" in result) {
      applySnapshot(result);
    }
    return result;
  } catch (error) {
    toast(error.message || String(error));
    return null;
  }
}
