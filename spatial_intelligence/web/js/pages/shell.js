/* Application shell: menubar, map panel, side panel, tabs, status bar.

The shell is rebuilt on every store change except that the map panel (and its
iframe) is mounted once per document, and the active tab keeps its scroll.
*/

"use strict";

import { renderStatusBar } from "../components/status-bar.js";
import { renderMenubar } from "../components/menubar.js";
import { el } from "../dom.js";
import { state, setState } from "../store.js";
import { hold, isFollowing, scrollTo, watchOutput } from "../scroll.js";
import { renderAddCellRow, renderCellsTab } from "./cells.js";
import { renderDataTab } from "./data.js";
import { mountMap, syncMap } from "./map.js";

const TABS = ["Cells", "Data"];

let rootEl = null;
let renderedWorkspace = null;
let tabScrollTop = 0;

export function mountShell(root) {
  rootEl = root || document.getElementById("app");
  watchOutput(rootEl);
  renderShell();
}

export function renderShell() {
  const root = shellRoot();
  if (!root) return;
  const menubar = root.querySelector("#menubar");
  const workspaceChanged = renderedWorkspace !== state.active_workspace;

  if (!menubar) {
    // First render: build the full shell. The map panel is created here and
    // re-appended (never rebuilt) by later renders, so the map cannot blink.
    root.replaceChildren(renderMenubar(), mountMap(), renderSidePanel());
    renderedWorkspace = state.active_workspace;
    return;
  }

  if (workspaceChanged) tabScrollTop = 0;
  else captureTabScroll();
  menubar.replaceWith(renderMenubar());
  const side = root.querySelector("#side-panel");
  if (side) side.replaceWith(renderSidePanel());
  syncMap();
  renderedWorkspace = state.active_workspace;
  if (!workspaceChanged) restoreTabScroll();
}

export function renderSidePanel() {
  const panel = el("div", { id: "side-panel" });
  const tabbar = el("div", { class: "tabbar" });
  for (const name of TABS) tabbar.append(tabButton(name));
  const content = el("div", { id: "tab-content" });
  content.append(renderTabContent());
  const toolbar =
    state.selected_tab === "Cells" && state.active_workspace ? renderAddCellRow() : null;
  appendToPanel(panel, tabbar, toolbar, content, renderStatusBar());
  return panel;
}

export function renderActiveTab() {
  const content = document.getElementById("tab-content");
  if (!content) return;
  captureTabScroll();
  content.replaceChildren(renderTabContent());
  restoreTabScroll();
}

export function renderTabContent() {
  return state.selected_tab === "Data" ? renderDataTab() : renderCellsTab();
}

function appendToPanel(panel, ...nodes) {
  panel.append(...nodes.filter((node) => node != null));
}

function tabButton(name) {
  return el("button", {
    text: name,
    class: name === state.selected_tab ? "active" : "",
    onclick: () => setState({ selected_tab: name }),
  });
}

function shellRoot() {
  return rootEl || document.getElementById("app");
}

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
