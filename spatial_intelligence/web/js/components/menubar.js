/* Menubar: workspace/file menu, cell menu, theme, settings, danger toggle. */

"use strict";

import { closeWorkspace, runAll, saveWorkspace, updateSettings } from "../api.js";
import { el, toast } from "../dom.js";
import { icon, menuLabel } from "../icons.js";
import { addCell } from "../pages/cells.js";
import { applySnapshot, setState, state } from "../store.js";
import { openSettingsDialog } from "./settings-dialog.js";
import { openWorkflowDialog } from "./workflow-dialog.js";
import { openWorkspaceDialog } from "./workspace-dialog.js";

let openMenu = null;

export function renderMenubar() {
  const bar = el("div", { id: "menubar" });
  bar.append(brand(), fileMenu(), cellMenu(), el("div", { class: "menubar-right" }, rightActions()));
  return bar;
}

export function closeAllMenus() {
  document.querySelectorAll(".menu.open").forEach((menu) => {
    menu.classList.remove("open");
    const button = menu.children[0];
    if (button) button.setAttribute("aria-expanded", "false");
  });
  openMenu = null;
}

function brand() {
  const wrap = el("div", { class: "brand" });
  const mark = el("span", { class: "brand-mark" });
  mark.append(icon("logo", { size: 21 }));
  const copy = el("div", { class: "brand-copy" });
  copy.append(el("span", { class: "brand-name", text: "Spatial Intelligence" }));
  wrap.append(mark, copy);
  return wrap;
}

function fileMenu() {
  const menu = el("div", { class: "menu" });
  const button = el("button", { text: "File", "aria-haspopup": "true", "aria-expanded": "false" });
  button.addEventListener("click", () => toggleMenu(menu));
  button.addEventListener("keydown", (event) => openMenuWithKey(menu, event));
  const items = el("div", { class: "menu-items", role: "menu" });
  items.append(
    menuItem("folder-plus", "New workspace…", () => openWorkflowDialog()),
    menuItem("folder-open", "Open workspace…", () => openWorkspaceDialog()),
    menuItem("save", "Save", doSave, "Ctrl S", !state.active_workspace),
    menuItem("x", "Close workspace", doClose, "", !state.active_workspace),
    el("div", { class: "separator" }),
    menuItem("settings", "Settings…", () => openSettingsDialog(), "Ctrl ,"),
    el("div", { class: "separator" }),
    menuItem("power", "Close all workspaces", doCloseAll)
  );
  menu.append(button, items);
  return menu;
}

function cellMenu() {
  const menu = el("div", { class: "menu" });
  const button = el("button", { text: "Cell", "aria-haspopup": "true", "aria-expanded": "false" });
  button.addEventListener("click", () => toggleMenu(menu));
  button.addEventListener("keydown", (event) => openMenuWithKey(menu, event));
  const items = el("div", { class: "menu-items", role: "menu" });
  items.append(
    menuItem("text", "Add Markdown Cell", () => runAction(() => addCell("markdown"))),
    menuItem("code", "Add Python Cell", () => runAction(() => addCell("python"))),
    menuItem("sparkles", "Add Prompt Cell", () => runAction(() => addCell("prompt"))),
    el("div", { class: "separator" }),
    menuItem("play", "Run All", () => runAction(() => runAll()), "Ctrl ⇧ Enter")
  );
  menu.append(button, items);
  return menu;
}

function rightActions() {
  const nodes = [];
  if (state.active_workspace) {
    const chip = el("button", {
      id: "ws-chip",
      type: "button",
      title: "Switch workspace",
      onclick: () => openWorkspaceDialog(),
    });
    chip.append(icon("folder", { size: 13 }), el("span", { text: state.active_workspace }));
    nodes.push(chip);
    nodes.push(el("span", { class: "menubar-divider" }));
  }
  nodes.push(themeToggle());
  nodes.push(
    iconButton("settings", "Settings", () => openSettingsDialog(), { id: "settings-button" })
  );
  nodes.push(dangerToggle());
  return nodes;
}

function themeToggle() {
  const dark = !!(state.settings && state.settings.theme === "dark");
  return iconButton(
    dark ? "sun" : "moon",
    dark ? "Switch to light theme" : "Switch to dark theme",
    async () => {
      try {
        const settings = await updateSettings({ theme: dark ? "light" : "dark" });
        setState({ settings });
      } catch (error) {
        toast(error.message || String(error), "error");
      }
    },
    { id: "theme-toggle" }
  );
}

function iconButton(name, title, onclick, options = {}) {
  const button = el("button", {
    class: "icon-btn",
    id: options.id || "",
    type: "button",
    title,
    "aria-label": title,
    onclick,
  });
  button.append(icon(name, { size: 17 }));
  return button;
}

function dangerToggle() {
  const on = !!(state.settings && state.settings.dangerous_mode);
  const title = on
    ? "Dangerous mode ON — run_python may execute arbitrary commands"
    : "Dangerous mode OFF — run_python is sandboxed";
  const button = el("button", {
    id: "danger-toggle",
    class: "danger-toggle" + (on ? " on" : ""),
    type: "button",
    title,
    "aria-pressed": String(on),
    "aria-label": title,
    onclick: () => toggleDangerous(!on),
  });
  button.append(
    icon(on ? "zap" : "shield", { size: 14 }),
    el("span", { class: "danger-label", text: on ? "Danger" : "Safe" })
  );
  return button;
}

function toggleMenu(menu) {
  const wasOpen = openMenu === menu;
  closeAllMenus();
  if (!wasOpen) {
    menu.classList.add("open");
    const button = menu.children[0];
    if (button) button.setAttribute("aria-expanded", "true");
    openMenu = menu;
  }
}

function openMenuWithKey(menu, event) {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  event.preventDefault();
  if (openMenu !== menu) toggleMenu(menu);
  const items = menuItems(menu);
  if (!items.length) return;
  (event.key === "ArrowDown" ? items[0] : items[items.length - 1]).focus();
}

function menuItems(menu) {
  return Array.from(menu.querySelectorAll(".menu-items button:not([disabled])"));
}

function moveMenuFocus(menu, button, event) {
  const items = menuItems(menu);
  const index = items.indexOf(button);
  if (event.key === "Escape") {
    event.preventDefault();
    closeAllMenus();
    const opener = menu.children[0];
    if (opener) opener.focus();
  } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const step = event.key === "ArrowDown" ? 1 : -1;
    items[(index + step + items.length) % items.length].focus();
  } else if (event.key === "Home" || event.key === "End") {
    event.preventDefault();
    items[event.key === "Home" ? 0 : items.length - 1].focus();
  } else if (event.key === "Tab") {
    closeAllMenus();
  }
}

function menuItem(name, label, onclick, shortcut = "", disabled = false) {
  const button = el("button", {
    type: "button",
    role: "menuitem",
    disabled: disabled ? "disabled" : null,
    onclick: () => {
      closeAllMenus();
      onclick();
    },
  });
  button.addEventListener("keydown", (event) => {
    const menu = button.closest(".menu");
    if (menu) moveMenuFocus(menu, button, event);
  });
  button.append(menuLabel(name, label, shortcut));
  return button;
}

async function toggleDangerous(next) {
  try {
    const settings = await updateSettings({ dangerous_mode: next });
    setState({ settings });
    toast(
      next
        ? "Dangerous mode enabled — run_python is no longer sandboxed"
        : "Dangerous mode disabled — run_python is sandboxed",
      next ? "warning" : "success"
    );
  } catch (error) {
    toast(error.message || String(error), "error");
  }
}

async function doSave() {
  if (!state.active_workspace) {
    toast("No workspace open", "warning");
    return;
  }
  try {
    await saveWorkspace();
    toast("Workspace saved", "success");
  } catch (error) {
    toast(error.message || String(error), "error");
  }
}

function doClose() {
  return runAction(() => closeWorkspace());
}

async function doCloseAll() {
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
    map_app_url: state.map_app_url,
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
    toast(error.message || String(error), "error");
    return null;
  }
}
