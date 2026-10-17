/* Status bar: active workspace, live-update state, and session token/cost totals. */

import { el, formatCost, formatTokens } from "../dom.js";
import { icon } from "../icons.js";
import { followToggle } from "../scroll.js";
import { state, usageTotals } from "../store.js";

let connection = "connecting";

export function setConnection(next) {
  const value = next === "live" || next === "reconnecting" ? next : "connecting";
  if (value === connection) return;
  connection = value;
  refreshStatusBar();
}

function connectionNode() {
  const labels = {
    live: "Live updates connected",
    reconnecting: "Live updates reconnecting…",
    connecting: "Connecting to live updates…",
  };
  const node = el("span", {
    class: "conn conn-" + connection,
    title: labels[connection],
    "aria-label": labels[connection],
    role: "status",
  });
  node.append(el("span", { class: "conn-dot" }));
  if (connection !== "live") {
    node.append(el("span", { text: connection === "reconnecting" ? "Reconnecting" : "Connecting" }));
  }
  return node;
}

export function renderStatusBar() {
  const bar = el("div", { id: "status-bar" });
  const active = state.active_workspace;
  const ws = el("span", { class: "status-ws" });
  ws.append(icon(active ? "folder" : "globe", { size: 13 }), el("span", { text: active || "No workspace" }));
  bar.append(ws, connectionNode());

  const right = el("div", { class: "status-right" });
  const u = usageTotals();
  if (u.has) {
    const stats = el("div", { class: "status-stats" });
    stats.append(
      el("span", { class: "stat", title: "Input tokens", text: "↑ " + formatTokens(u.input) }),
      el("span", { class: "stat", title: "Output tokens", text: "↓ " + formatTokens(u.output) }),
      el("span", { class: "stat", title: "Model requests", text: u.requests + " req" })
    );
    const cost = formatCost(u.cost);
    if (cost) stats.append(el("span", { class: "stat cost", text: cost }));
    const ratio = u.input ? Math.round((u.cacheRead / u.input) * 100) : 0;
    stats.append(
      el("span", { class: "stat", title: "Prompt cache hit ratio (DeepSeek)", text: ratio + "% cached" })
    );
    right.append(stats);
  }
  right.append(followToggle());
  bar.append(right);
  return bar;
}

export function refreshStatusBar() {
  const bar = document.getElementById("status-bar");
  if (bar && bar.parentElement) {
    bar.replaceWith(renderStatusBar());
  }
}
