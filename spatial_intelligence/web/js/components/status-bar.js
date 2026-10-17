/* Status bar: live-update state, session token/cost totals, and Follow output.

The workspace name lives in the menubar chip, so it is not repeated here. The
connection state is a coloured dot with a hover tooltip; request/cache details
live in the token summary's tooltip, keeping the bar to one line in a narrow
panel.
*/

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
  return node;
}

function visionNode() {
  const vision = state.vision;
  if (!vision || !vision.available) return null;
  const label =
    "Image input available" + (vision.model ? " · " + vision.model : "");
  const node = el("span", {
    class: "vision-status",
    role: "status",
    title: label,
    "aria-label": label,
  });
  node.append(icon("eye", { size: 15 }));
  return node;
}

function usageDetails(u) {
  const rounded = u.input ? Math.round((u.cacheRead / u.input) * 100) : 0;
  const parts = [
    u.requests + (u.requests === 1 ? " request" : " requests"),
    rounded + "% cached",
  ];
  if (u.toolCalls) {
    parts.push(u.toolCalls + (u.toolCalls === 1 ? " tool call" : " tool calls"));
  }
  return parts.join(" · ");
}

export function renderStatusBar() {
  const bar = el("div", { id: "status-bar" });
  bar.append(connectionNode());
  const vision = visionNode();
  if (vision) bar.append(vision);

  const u = usageTotals();
  if (u.has) {
    const stats = el("div", {
      class: "status-stats",
      title: usageDetails(u),
      "aria-label": "Token usage: " + usageDetails(u),
    });
    const ratio = u.input ? Math.round((u.cacheRead / u.input) * 100) : 0;
    stats.append(
      el("span", { class: "stat", text: "↑ " + formatTokens(u.input) }),
      el("span", { class: "stat", text: "↓ " + formatTokens(u.output) }),
      el("span", { class: "stat", title: "Model requests", text: u.requests + " req" }),
      el("span", {
        class: "stat cache",
        title: "Prompt cache hit rate",
        text: ratio + "% cache",
      })
    );
    const cost = formatCost(u.cost);
    if (cost) stats.append(el("span", { class: "stat cost", text: cost }));
    bar.append(stats);
  }

  const right = el("div", { class: "status-right" });
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
