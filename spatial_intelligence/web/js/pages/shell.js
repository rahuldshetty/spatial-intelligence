/* Application shell: menubar, map panel, resizable splitter, side panel, status.

The shell is rebuilt on every store change except that the map panel (and its
iframe) is mounted once per document, and the active tab keeps its scroll.
*/

"use strict";

import { renderStatusBar } from "../components/status-bar.js";
import { renderMenubar } from "../components/menubar.js";
import { openWorkflowDialog } from "../components/workflow-dialog.js";
import { openWorkspaceDialog } from "../components/workspace-dialog.js";
import { el } from "../dom.js";
import { icon } from "../icons.js";
import { state, setState } from "../store.js";
import { hold, isFollowing, scrollTo, watchOutput } from "../scroll.js";
import { renderAddCellRow, renderCellsTab } from "./cells.js";
import { manageableWorkspaceFiles, renderDataTab } from "./data.js";
import { mountMap, syncMap } from "./map.js";

const TABS = [
  { name: "Cells", icon: "notebook" },
  { name: "Data", icon: "database" },
];
const PANEL_WIDTH_KEY = "spatial.panelWidth";
const PANEL_MIN = 320;
const PANEL_MAX = 900;
const PANEL_DEFAULT = 432;

let rootEl = null;
let renderedWorkspace = null;
let tabScrollTop = 0;
let splitterEl = null;
let panelWidth = PANEL_DEFAULT;
let welcomeEl = null;

export function mountShell(root) {
  rootEl = root || document.getElementById("app");
  watchOutput(rootEl);
  renderShell();
}

export function renderShell() {
  const root = shellRoot();
  if (!root) return;
  document.title = state.active_workspace
    ? state.active_workspace + " — Spatial Intelligence"
    : "Spatial Intelligence";
  const menubar = root.querySelector("#menubar");
  const workspaceChanged = renderedWorkspace !== state.active_workspace;

  if (!menubar) {
    // First render: build the full shell. The map panel is created here and
    // re-appended (never rebuilt) by later renders, so the map cannot blink.
    root.replaceChildren(
      renderMenubar(),
      mountMap(),
      mountSplitter(),
      renderSidePanel(),
      mountWelcome()
    );
    syncWelcome();
    renderedWorkspace = state.active_workspace;
    return;
  }

  if (workspaceChanged) tabScrollTop = 0;
  else captureTabScroll();
  menubar.replaceWith(renderMenubar());
  if (!root.querySelector("#splitter")) {
    root.append(mountSplitter());
  }
  if (!root.querySelector("#welcome")) {
    root.append(mountWelcome());
  }
  const side = root.querySelector("#side-panel");
  if (side) side.replaceWith(renderSidePanel());
  syncWelcome();
  syncMap();
  renderedWorkspace = state.active_workspace;
  if (!workspaceChanged) restoreTabScroll();
}

export function renderSidePanel() {
  const panel = el("div", { id: "side-panel" });
  const tabbar = el("div", { class: "tabbar", role: "tablist" });
  for (const tab of TABS) tabbar.append(tabButton(tab));
  const content = el("div", { id: "tab-content" });
  content.append(renderTabContent());
  const toolbar =
    state.selected_tab === "Cells" && state.active_workspace
      ? el("div", { class: "panel-toolbar" }, [renderAddCellRow()])
      : null;
  appendToPanel(panel, tabbar, toolbar, content, renderStatusBar());
  return panel;
}

export function renderTabContent() {
  return state.selected_tab === "Data" ? renderDataTab() : renderCellsTab();
}

function appendToPanel(panel, ...nodes) {
  panel.append(...nodes.filter((node) => node != null));
}

function tabButton(tab) {
  const active = tab.name === state.selected_tab;
  const button = el("button", {
    class: active ? "active" : "",
    role: "tab",
    "aria-selected": String(active),
  });
  button.append(icon(tab.icon, { size: 15 }), el("span", { text: tab.name }));
  const count =
    tab.name === "Data"
      ? manageableWorkspaceFiles(state.files).length
      : state.cells.length;
  if (count) button.append(el("span", { class: "tab-count", text: String(count) }));
  button.addEventListener("click", () => setState({ selected_tab: tab.name }));
  return button;
}

function shellRoot() {
  return rootEl || document.getElementById("app");
}

/* -- welcome -------------------------------------------------------------- */

function mountWelcome() {
  if (welcomeEl) return welcomeEl;
  welcomeEl = el("div", { id: "welcome" });
  const card = el("section", { class: "welcome-card", "aria-label": "Start a workspace" });
  const mark = el("span", { class: "welcome-mark" });
  mark.append(icon("logo", { size: 36 }));
  card.append(
    mark,
    el("h2", { text: "Start a workspace" }),
    el("p", {
      text:
        "Ask a geospatial question in plain language. The agent finds the imagery, runs the analysis, and puts the result on the live map.",
    })
  );
  const actions = el("div", { class: "welcome-actions" });
  actions.append(
    el("button", {
      class: "primary",
      type: "button",
      text: "New workspace",
      onclick: () => openWorkflowDialog(),
    }),
    el("button", {
      class: "secondary",
      type: "button",
      text: "Open workspace…",
      onclick: () => openWorkspaceDialog(),
    })
  );
  const features = el("ul", { class: "welcome-features" });
  for (const [name, label] of [
    ["layers", "Find imagery"],
    ["code", "Run analysis"],
    ["globe", "Map results"],
    ["notebook", "Keep every result"],
  ]) {
    const item = el("li", {});
    item.append(icon(name, { size: 15 }), el("span", { text: label }));
    features.append(item);
  }
  card.append(actions, features);
  welcomeEl.append(card);
  return welcomeEl;
}

function syncWelcome() {
  if (welcomeEl) welcomeEl.hidden = Boolean(state.active_workspace);
}

/* -- splitter ------------------------------------------------------------- */

function mountSplitter() {
  if (splitterEl) return splitterEl;
  splitterEl = el("div", {
    id: "splitter",
    role: "separator",
    "aria-orientation": "vertical",
    "aria-label": "Resize map and notebook panels",
    tabindex: "0",
    title: "Drag to resize · double-click to reset",
  });
  splitterEl.append(el("span", { class: "splitter-grip" }));
  splitterEl.addEventListener("pointerdown", startResize);
  splitterEl.addEventListener("dblclick", () => setPanelWidth(PANEL_DEFAULT, true));
  splitterEl.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setPanelWidth(panelWidth + 24, true);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setPanelWidth(panelWidth - 24, true);
    } else if (event.key === "Home") {
      event.preventDefault();
      setPanelWidth(PANEL_DEFAULT, true);
    }
  });
  const stored = Number(readStoredWidth());
  setPanelWidth(stored || defaultPanelWidth(), false);
  return splitterEl;
}

function startResize(event) {
  if (event.button !== 0) return;
  event.preventDefault();
  document.body.classList.add("resizing");
  const move = (moveEvent) => setPanelWidth(window.innerWidth - moveEvent.clientX, false);
  const stop = () => {
    document.body.classList.remove("resizing");
    window.removeEventListener("pointermove", move);
    rememberWidth();
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop, { once: true });
}

function defaultPanelWidth() {
  const viewport = Number(window.innerWidth) || 1440;
  return Math.round(Math.min(560, Math.max(380, viewport * 0.3)));
}

function setPanelWidth(width, persist) {
  const viewport = Number(window.innerWidth) || 1440;
  const maximum = Math.min(PANEL_MAX, Math.max(PANEL_MIN, viewport - 360));
  panelWidth = Math.round(Math.min(maximum, Math.max(PANEL_MIN, Number(width) || PANEL_DEFAULT)));
  if (splitterEl) {
    splitterEl.setAttribute("aria-valuemin", String(PANEL_MIN));
    splitterEl.setAttribute("aria-valuemax", String(maximum));
    splitterEl.setAttribute("aria-valuenow", String(panelWidth));
  }
  const style = document.documentElement && document.documentElement.style;
  if (style) {
    if (typeof style.setProperty === "function") style.setProperty("--panel-width", panelWidth + "px");
    else style["--panel-width"] = panelWidth + "px";
  }
  if (persist) rememberWidth();
}

function readStoredWidth() {
  try {
    return window.localStorage ? window.localStorage.getItem(PANEL_WIDTH_KEY) : null;
  } catch (_) {
    return null;
  }
}

function rememberWidth() {
  try {
    if (window.localStorage) window.localStorage.setItem(PANEL_WIDTH_KEY, String(panelWidth));
  } catch (_) {
    /* private mode or a storage quota: the size still applies this session */
  }
}

/* -- tab scroll ----------------------------------------------------------- */

function captureTabScroll() {
  const content = document.getElementById("tab-content");
  tabScrollTop = content ? content.scrollTop : 0;
}

function restoreTabScroll() {
  const content = document.getElementById("tab-content");
  if (!content) return;
  if (isFollowing()) {
    // Following wants the bottom, so a remembered position is overruled.
    hold();
    return;
  }
  const target = Math.min(tabScrollTop, Math.max(0, content.scrollHeight - content.clientHeight));
  requestAnimationFrame(() => {
    scrollTo(content, target);
  });
}
