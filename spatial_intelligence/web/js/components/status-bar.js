/* Status bar: active workspace plus per-session token/cost totals. */

import { el, formatCost, formatTokens } from "../dom.js";
import { followToggle } from "../scroll.js";
import { state, usageTotals } from "../store.js";

export function renderStatusBar() {
  const bar = el("div", { id: "status-bar" });
  const ws = el("span", { class: "status-ws", text: state.active_workspace || "No workspace" });
  bar.append(ws);
  const right = el("div", { class: "status-right" });
  const u = usageTotals();
  if (u.has) {
    const stats = el("div", { class: "status-stats" });
    stats.append(
      el("span", { class: "stat", title: "Input tokens", text: "↑ " + formatTokens(u.input) }),
      el("span", { class: "stat", title: "Output tokens", text: "↓ " + formatTokens(u.output) }),
      el("span", { class: "stat", title: "Requests", text: u.requests + " req" })
    );
    const cost = formatCost(u.cost);
    if (cost) stats.append(el("span", { class: "stat cost", text: cost }));
    const ratio = u.input ? Math.round((u.cacheRead / u.input) * 100) : 0;
    stats.append(el("span", { class: "stat", title: "Prompt cache hit ratio (DeepSeek)", text: ratio + "% cached" }));
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
