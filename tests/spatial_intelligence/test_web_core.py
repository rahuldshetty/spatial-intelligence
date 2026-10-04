"""Frontend core (ES modules) tests.

Runs the real modules under Node with a minimal DOM shim: element/selector
shim, fake EventSource and fetch. Every assertion observes behaviour a
consumer sees — returned values, DOM the module built, requests it issued,
messages it posted — never module internals.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[2]

HARNESS = r"""
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(process.argv[2]);
const WEB = path.join(ROOT, "spatial_intelligence", "web", "js");
const webUrl = (rel) => new URL("file:///" + path.join(WEB, rel).replace(/\\/g, "/")).href;
const realSetTimeout = globalThis.setTimeout;
const tick = () => new Promise((resolve) => realSetTimeout(resolve, 0));

const checks = {};
const details = {};
function check(name, ok, detail) {
  checks[name] = Boolean(ok);
  if (!ok) details[name] = detail === undefined ? "" : String(detail);
}
function eq(name, actual, expected) {
  check(
    name,
    JSON.stringify(actual) === JSON.stringify(expected),
    "actual=" + JSON.stringify(actual) + " expected=" + JSON.stringify(expected)
  );
}
async function rejects(fn) {
  try {
    await fn();
    return null;
  } catch (error) {
    return error.message;
  }
}
async function scenario(name, fn) {
  try {
    await fn();
  } catch (error) {
    check(name + ".threw", false, error && error.stack ? error.stack : String(error));
  }
}

// -- DOM shim ---------------------------------------------------------------

function textNode(value) {
  return { nodeType: 3, children: [], parentNode: null, textContent: String(value) };
}

function parseCompound(text) {
  const compound = { tag: null, id: null, classes: [], attrs: [] };
  const pattern = /([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:=["']?([^\]"']*)["']?)?\]/g;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match[1]) compound.tag = match[1].toUpperCase();
    else if (match[2]) compound.id = match[2];
    else if (match[3]) compound.classes.push(match[3]);
    else if (match[4]) compound.attrs.push([match[4], match[5] === undefined ? null : match[5]]);
  }
  return compound;
}

function matchesCompound(node, compound) {
  if (node.nodeType !== 1) return false;
  if (compound.tag && node.tagName !== compound.tag) return false;
  if (compound.id && node.getAttribute("id") !== compound.id) return false;
  for (const name of compound.classes) if (!node.classes.has(name)) return false;
  for (const [name, value] of compound.attrs) {
    if (!node.hasAttribute(name)) return false;
    if (value !== null && node.getAttribute(name) !== value) return false;
  }
  return true;
}

function matchesSelector(node, selector) {
  const chain = String(selector).trim().split(/\s+/).map(parseCompound);
  let index = chain.length - 1;
  if (!matchesCompound(node, chain[index])) return false;
  index -= 1;
  let current = node.parentNode;
  while (index >= 0) {
    let found = false;
    while (current) {
      if (current.nodeType === 1 && matchesCompound(current, chain[index])) {
        found = true;
        current = current.parentNode;
        break;
      }
      current = current.parentNode;
    }
    if (!found) return false;
    index -= 1;
  }
  return true;
}

class ShimNode {
  constructor(tag) {
    this.nodeType = 1;
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.attributes = new Map();
    this.dataset = {};
    this.style = {};
    this.listeners = new Map();
    this.classes = new Set();
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.readOnly = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    if (this.tagName === "IFRAME") {
      const self = this;
      this.contentWindow = {
        postMessage(message, target) {
          posted.push({ message, target, frame: self });
        },
      };
    }
  }
  get className() {
    return [...this.classes].join(" ");
  }
  set className(value) {
    this.classes = new Set(String(value).split(/\s+/).filter(Boolean));
  }
  get classList() {
    const self = this;
    return {
      add: (...names) => names.forEach((name) => self.classes.add(name)),
      remove: (...names) => names.forEach((name) => self.classes.delete(name)),
      contains: (name) => self.classes.has(name),
      toggle: (name, force) => {
        const on = force === undefined ? !self.classes.has(name) : Boolean(force);
        if (on) self.classes.add(name);
        else self.classes.delete(name);
        return on;
      },
    };
  }
  get textContent() {
    return this.children.map((child) => child.textContent).join("");
  }
  set textContent(value) {
    this.children.forEach((child) => {
      child.parentNode = null;
    });
    this.children = [textNode(value)];
  }
  get innerHTML() {
    return this.children.map((child) => child.textContent).join("");
  }
  set innerHTML(value) {
    this.children.forEach((child) => {
      child.parentNode = null;
    });
    this.children = value ? [textNode(value)] : [];
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  append(...nodes) {
    for (const raw of nodes) {
      if (raw === null || raw === undefined) continue;
      const node = typeof raw === "string" ? textNode(raw) : raw;
      if (node.parentNode && node.parentNode !== this) {
        const index = node.parentNode.children.indexOf(node);
        if (index >= 0) node.parentNode.children.splice(index, 1);
      }
      node.parentNode = this;
      this.children.push(node);
    }
  }
  prepend(node) {
    node.parentNode = this;
    this.children.unshift(node);
  }
  remove() {
    if (!this.parentNode) return;
    const index = this.parentNode.children.indexOf(this);
    if (index >= 0) this.parentNode.children.splice(index, 1);
    this.parentNode = null;
  }
  replaceChildren(...nodes) {
    this.children.forEach((child) => {
      child.parentNode = null;
    });
    this.children = [];
    this.append(...nodes);
  }
  replaceWith(node) {
    const parent = this.parentNode;
    if (!parent) return;
    const index = parent.children.indexOf(this);
    if (index < 0) return;
    this.parentNode = null;
    node.parentNode = parent;
    parent.children.splice(index, 1, node);
  }
  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(handler);
  }
  dispatch(type, event = {}) {
    for (const handler of this.listeners.get(type) || []) handler(event);
  }
  click() {
    this.dispatch("click", { target: this });
  }
  focus() {}
  scrollIntoView() {}
  get lastElementChild() {
    for (let index = this.children.length - 1; index >= 0; index -= 1) {
      if (this.children[index].nodeType === 1) return this.children[index];
    }
    return null;
  }
  get parentElement() {
    return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null;
  }
  get isConnected() {
    let node = this;
    while (node) {
      if (node.connected) return true;
      node = node.parentNode;
    }
    return false;
  }
  descendants(out) {
    for (const child of this.children) {
      if (child.nodeType !== 1) continue;
      out.push(child);
      child.descendants(out);
    }
    return out;
  }
  querySelectorAll(selector) {
    return this.descendants([]).filter((node) => matchesSelector(node, selector));
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  closest(selector) {
    let node = this;
    while (node) {
      if (node.nodeType === 1 && matchesSelector(node, selector)) return node;
      node = node.parentNode;
    }
    return null;
  }
}

const documentElement = new ShimNode("html");
documentElement.connected = true;
const body = new ShimNode("body");
body.connected = true;
documentElement.append(body);

const documentListeners = new Map();
const document = {
  readyState: "complete",
  documentElement,
  body,
  createElement: (tag) => new ShimNode(tag),
  createTextNode: (value) => textNode(value),
  getElementById: (id) => documentElement.descendants([]).find((node) => node.getAttribute("id") === id) || null,
  querySelector: (selector) => documentElement.querySelector(selector),
  querySelectorAll: (selector) => documentElement.querySelectorAll(selector),
  addEventListener(type, handler) {
    if (!documentListeners.has(type)) documentListeners.set(type, []);
    documentListeners.get(type).push(handler);
  },
};

const posted = [];
const messageHandlers = [];
const eventSources = [];
const fetchCalls = [];
let fetchResponder = () => ({ ok: true, status: 200, statusText: "OK", json: async () => ({}) });

globalThis.window = globalThis;
globalThis.document = document;
globalThis.addEventListener = (type, handler) => {
  if (type === "message") messageHandlers.push(handler);
};
globalThis.requestAnimationFrame = (callback) => {
  callback(0);
  return 0;
};
globalThis.fetch = async (url, options = {}) => {
  fetchCalls.push({
    url: String(url),
    method: options.method || "GET",
    body: options.body,
    headers: options.headers,
  });
  return fetchResponder(String(url), options);
};
globalThis.EventSource = class EventSource {
  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    this.closed = false;
    eventSources.push(this);
  }
  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(handler);
  }
  close() {
    this.closed = true;
  }
  emit(type, raw) {
    for (const handler of this.listeners.get(type) || []) handler({ data: raw });
  }
};

function sendMessage(origin, source, data) {
  for (const handler of messageHandlers) handler({ origin, source, data });
}

// -- scenarios --------------------------------------------------------------

await scenario("dom", async () => {
  const dom = await import(webUrl("dom.js"));

  const clicks = [];
  const node = dom.el(
    "div",
    { class: "card wide", text: "hi", "data-kind": "cell", value: null, onclick: () => clicks.push("clicked") },
    [dom.el("span", { text: "kid" }), "tail"]
  );
  eq("dom.class_attr", node.className, "card wide");
  eq("dom.text_before_children", node.textContent, "hikidtail");
  eq("dom.set_attribute", node.getAttribute("data-kind"), "cell");
  eq("dom.null_attribute_skipped", node.hasAttribute("value"), false);
  node.dispatch("click", { target: node });
  eq("dom.listener", clicks, ["clicked"]);

  const list = dom.el("ul", {});
  dom.appendChildren(list, dom.el("li", { text: "a" }), null, "raw");
  eq("dom.append_children_skips_null", list.children.filter((child) => child.nodeType === 1).length, 1);
  eq("dom.append_children_text", list.textContent, "araw");

  eq("dom.format_bytes_zero", dom.formatBytes(0), "0 B");
  eq("dom.format_bytes_negative", dom.formatBytes(-5), "0 B");
  eq("dom.format_bytes_kb", dom.formatBytes(2048), "2 KB");
  eq("dom.format_bytes_mb", dom.formatBytes(5 * 1024 * 1024), "5.0 MB");
  eq("dom.format_tokens_k", dom.formatTokens(1500), "1.5k");
  eq("dom.format_tokens_m", dom.formatTokens(2500000), "2.5M");
  eq("dom.format_cost_zero", dom.formatCost(0), null);
  eq("dom.format_cost_value", dom.formatCost(0.25), "$0.2500");
  eq("dom.usage_label", dom.usageLabel({ input_tokens: 1500, output_tokens: 200, cost: 0.25 }), "↑1.5k ↓200 $0.2500");
  eq(
    "dom.usage_title",
    dom.usageTitle({ requests: 1, tool_calls: 2, total_tokens: 5 }),
    "1 request · 2 tool calls · 5 total tokens"
  );
  eq("dom.truncate", dom.truncate("abcdef", 3), "abc…");
  eq("dom.pretty_json", dom.prettyValue('{"b":1,"a":2}'), '{\n  "b": 1,\n  "a": 2\n}');
  eq("dom.pretty_raw", dom.prettyValue("not json"), "not json");
  eq("dom.compact_preview", dom.compactPreview("  a\n\n b "), "a b");

  const block = dom.codeBlock("print(1)");
  eq("dom.code_block", [block.tagName, block.textContent], ["PRE", "print(1)"]);
  const expansion = dom.expandableContent("Input", "abc", "error");
  eq("dom.expandable_class", expansion.className, "agent-action-content error");
  eq("dom.expandable_pre", expansion.querySelector("pre").textContent, "abc");
  eq("dom.expandable_summary", expansion.querySelector("summary").querySelector("strong").textContent, "Input");

  const timers = [];
  globalThis.setTimeout = (fn, ms) => {
    timers.push(ms);
    return {};
  };
  try {
    dom.toast("first");
    const toastNode = document.getElementById("toast");
    eq("dom.toast_text", toastNode.textContent, "first");
    eq("dom.toast_visible", toastNode.style.display, "block");
    dom.toast("second");
    eq("dom.toast_single_node", document.querySelectorAll("#toast").length, 1);
    eq("dom.toast_updated", toastNode.textContent, "second");
    eq("dom.toast_timer_reset", timers.length, 2);
    check("dom.toast_timer_not_on_node", toastNode._timer === undefined, String(toastNode._timer));
  } finally {
    globalThis.setTimeout = realSetTimeout;
  }
});

await scenario("store", async () => {
  const store = await import(webUrl("store.js"));

  store.setState({ selected_tab: "Cells" });
  const seen = [];
  const off = store.subscribe((snapshot) => seen.push(snapshot.selected_tab));
  store.setState({ selected_tab: "Data" });
  off();
  store.setState({ selected_tab: "Cells" });
  eq("store.subscribe_notify", seen, ["Data"]);

  check("store.generated_flag", store.isGeneratedCell({ metadata: { geoai: { generated: true } } }) === true);
  check("store.generated_interaction", store.isGeneratedCell({ metadata: { geoai: { kind: "interaction" } } }) === true);
  check("store.generated_plain", store.isGeneratedCell({ metadata: {} }) === false);

  store.setState({
    cells: [
      { id: "c1", kind: "python", source: 42 },
      { id: "g", kind: "tool", metadata: { geoai: { generated: true } } },
      null,
    ],
  });
  eq("store.normalize_drops_generated", store.state.cells.map((cell) => cell.id), ["c1"]);
  eq("store.normalize_source_string", store.state.cells[0].source, "42");

  store.setState({ files: ["a.tif", "", null, 3] });
  eq("store.files_filtered", store.state.files, ["a.tif"]);

  store.setState({
    jobs: [
      { job_id: "j1", parent_id: "c1", kind: "download" },
      { job_id: "j2", parent_id: "c2", kind: "raster" },
      { job_id: "j3", parent_id: "c1", kind: "download" },
    ],
  });
  eq("store.jobs_for_any_kind", store.jobsFor("c1").map((job) => job.job_id), ["j1", "j3"]);
  eq("store.jobs_for_other", store.jobsFor("c2").map((job) => job.job_id), ["j2"]);
  eq("store.jobs_for_missing", store.jobsFor("nope"), []);

  store.upsertJob({ job_id: "j1", status: "running", completed: 10, total: 100 });
  const merged = store.upsertJob({ job_id: "j1", status: "done", completed: 100 });
  eq("store.upsert_merges", [merged.kind, merged.status, merged.completed, merged.total], ["download", "done", 100, 100]);
  eq("store.upsert_no_duplicate", store.state.jobs.filter((job) => job.job_id === "j1").length, 1);

  store.setState({
    cells: [
      { id: "a", kind: "prompt", usage: { input_tokens: 100, output_tokens: 20, requests: 1, tool_calls: 2, cost: 0.5, cache_read_tokens: 40 } },
      { id: "b", kind: "prompt", usage: { input_tokens: 10, output_tokens: 5, requests: 1, cost: null } },
      { id: "c", kind: "python" },
    ],
  });
  const totals = store.usageTotals();
  eq(
    "store.usage_totals",
    [totals.input, totals.output, totals.requests, totals.toolCalls, totals.cost, totals.cacheRead, totals.has],
    [110, 25, 2, 2, 0.5, 40, true]
  );
  store.setState({ cells: [] });
  eq("store.usage_empty", store.usageTotals().has, false);

  // A snapshot is built before its response arrives: a running cell keeps the
  // steps streamed while the request was in flight instead of dropping them.
  store.setState({
    cells: [
      {
        id: "live",
        kind: "prompt",
        status: "running",
        trace: [
          { type: "text_delta", content: "Downloading." },
          { type: "tool_call", name: "download", tool_call_id: "t1" },
          { type: "tool_result", name: "download", tool_call_id: "t1", content: "f.geojson" },
        ],
      },
      { id: "done", kind: "prompt", status: "running", trace: [{ type: "text", content: "old" }] },
    ],
  });
  store.applySnapshot({
    active_workspace: "ws",
    cells: [
      {
        id: "live",
        kind: "prompt",
        status: "running",
        trace: [
          { type: "text_delta", content: "Downloading." },
          { type: "tool_call", name: "download", tool_call_id: "t1" },
        ],
      },
      { id: "done", kind: "prompt", status: "done", trace: [] },
      { id: "fresh", kind: "prompt", status: "idle", trace: [] },
    ],
  });
  const kept = store.state.cells.find((cell) => cell.id === "live");
  eq("store.snapshot_keeps_streamed_tail", kept.trace.length, 3);
  eq("store.snapshot_starts_snapshot_steps", kept.trace.map((step) => step.type), [
    "text_delta",
    "tool_call",
    "tool_result",
  ]);
  eq("store.snapshot_replaces_a_finished_cell", store.state.cells.find((cell) => cell.id === "done").trace, []);
  eq("store.snapshot_adds_new_cells", store.state.cells.map((cell) => cell.id), ["live", "done", "fresh"]);

  store.applySnapshot({
    active_workspace: "ws",
    workspaces: ["ws"],
    cells: [{ id: "g", metadata: { geoai: { generated: true } } }],
    map_project: null,
    map_app_url: "http://127.0.0.1:8100/",
    files: ["x.tif"],
    jobs: [{ job_id: "j", parent_id: "c" }],
    settings: { model: "m", theme: "dark", dangerous_mode: true, max_retries: 3, record_agent_steps: false },
  });
  check("store.snapshot_workspace", store.state.active_workspace === "ws");
  eq("store.snapshot_drops_generated", store.state.cells.length, 0);
  eq("store.snapshot_files", store.state.files, ["x.tif"]);
  eq("store.snapshot_jobs", store.state.jobs.length, 1);
  eq("store.snapshot_theme_attr", document.documentElement.getAttribute("data-theme"), "dark");
  eq("store.snapshot_keeps_tab", store.state.selected_tab, "Cells");
});

await scenario("api", async () => {
  const api = await import(webUrl("api.js"));
  const store = await import(webUrl("store.js"));
  fetchResponder = () => ({ ok: true, status: 200, statusText: "OK", json: async () => ({ ok: true }) });

  fetchCalls.length = 0;
  await api.getState();
  eq("api.get_state", [fetchCalls[0].url, fetchCalls[0].method], ["/api/state", "GET"]);

  fetchCalls.length = 0;
  await api.addCell("python");
  eq(
    "api.add_cell",
    [fetchCalls[0].url, fetchCalls[0].method, fetchCalls[0].body],
    ["/api/cells", "POST", JSON.stringify({ kind: "python", source: "", index: null })]
  );

  fetchCalls.length = 0;
  await api.updateCell("c1", "x = 1");
  eq(
    "api.update_cell",
    [fetchCalls[0].url, fetchCalls[0].method, fetchCalls[0].body],
    ["/api/cells/c1", "PUT", JSON.stringify({ source: "x = 1" })]
  );

  store.setState({ cells: [{ id: "a" }, { id: "b" }] });
  fetchCalls.length = 0;
  await api.moveCell("a", 1);
  eq("api.move_cell", [fetchCalls[0].url, fetchCalls[0].body], ["/api/cells/a/move", JSON.stringify({ index: 1 })]);
  fetchCalls.length = 0;
  eq("api.move_cell_out_of_range", await api.moveCell("b", 1), null);
  eq("api.move_cell_no_request", fetchCalls.length, 0);

  fetchCalls.length = 0;
  await api.respondInteraction("c1", "i1", { f: 1 });
  eq(
    "api.respond_interaction",
    [fetchCalls[0].url, fetchCalls[0].body],
    ["/api/cells/c1/interaction", JSON.stringify({ interaction_id: "i1", answers: { f: 1 } })]
  );

  fetchCalls.length = 0;
  await api.cancelInteraction("c1", "i1");
  eq("api.cancel_interaction", [fetchCalls[0].url, fetchCalls[0].method], ["/api/cells/c1/interaction/i1", "DELETE"]);

  fetchCalls.length = 0;
  await api.updateSettings({ theme: "dark" });
  eq(
    "api.update_settings",
    [fetchCalls[0].url, fetchCalls[0].method, fetchCalls[0].body],
    ["/api/settings", "PUT", JSON.stringify({ theme: "dark" })]
  );

  fetchCalls.length = 0;
  await api.importUrl("https://example.com/a.tif");
  eq(
    "api.import_url",
    [fetchCalls[0].url, fetchCalls[0].body],
    ["/api/import/url", JSON.stringify({ url: "https://example.com/a.tif", filename: null })]
  );

  fetchCalls.length = 0;
  await api.reportBridge({ version: "1.2", methods: ["project.load"] });
  eq(
    "api.report_bridge",
    [fetchCalls[0].url, fetchCalls[0].body],
    ["/api/geolibre/bridge", JSON.stringify({ version: "1.2", methods: ["project.load"] })]
  );

  fetchCalls.length = 0;
  await api.setMapProject({ layers: [] });
  eq("api.set_map_project", [fetchCalls[0].url, fetchCalls[0].body], ["/api/map/project", JSON.stringify({ project: { layers: [] } })]);

  fetchCalls.length = 0;
  await api.closeWorkspace();
  eq("api.close_workspace", [fetchCalls[0].url, fetchCalls[0].method], ["/api/workspace/close", "POST"]);
  fetchCalls.length = 0;
  await api.runAll();
  eq("api.run_all", [fetchCalls[0].url, fetchCalls[0].method], ["/api/run-all", "POST"]);
  fetchCalls.length = 0;
  await api.stopCell("c1");
  eq("api.stop_cell", [fetchCalls[0].url, fetchCalls[0].method], ["/api/cells/c1/stop", "POST"]);

  const file = new File(["data"], "x.tif");
  file.webkitRelativePath = "folder/x.tif";
  fetchCalls.length = 0;
  await api.importFiles([file]);
  const upload = fetchCalls[0];
  check(
    "api.import_files_request",
    upload.url === "/api/import/local" && upload.method === "POST" && upload.body instanceof FormData,
    upload.url + " " + upload.method
  );
  const uploaded = upload.body.get("files");
  check("api.import_files_name", uploaded && uploaded.name === "folder/x.tif", uploaded && uploaded.name);

  fetchResponder = () => ({ ok: false, status: 404, statusText: "Not Found", json: async () => ({ detail: "cell not found" }) });
  eq("api.error_detail", await rejects(() => api.deleteCell("nope")), "cell not found");
  fetchResponder = () => ({ ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: { a: 1 } }) });
  eq("api.error_object_detail", await rejects(() => api.getState()), JSON.stringify({ a: 1 }));
  fetchResponder = () => ({
    ok: false,
    status: 500,
    statusText: "Server Error",
    json: async () => {
      throw new Error("no body");
    },
  });
  eq("api.error_status_text", await rejects(() => api.getState()), "Server Error");
  fetchResponder = () => ({ ok: true, status: 200, statusText: "OK", json: async () => ({}) });
});

await scenario("events", async () => {
  const events = await import(webUrl("events.js"));
  const scheduled = [];
  globalThis.setTimeout = (fn, ms) => {
    scheduled.push({ fn, ms });
    return {};
  };
  try {
    const received = [];
    const stop = events.connectEvents({
      onCell: (payload) => received.push(["cell", payload]),
      onTrace: (payload) => received.push(["trace", payload]),
      onJob: (payload) => received.push(["job", payload]),
      onJobs: (payload) => received.push(["jobs", payload]),
      onMap: (payload) => received.push(["map", payload]),
      onFiles: (payload) => received.push(["files", payload]),
      onSettings: (payload) => received.push(["settings", payload]),
    });

    eq("events.single_source", eventSources.length, 1);
    eq("events.url", eventSources[0].url, "/api/events");
    const source = eventSources[0];
    eq(
      "events.registered_events",
      [...source.listeners.keys()].sort(),
      ["cell", "files", "job", "jobs", "map", "open", "settings", "trace"]
    );

    for (const name of ["cell", "trace", "job", "jobs", "map", "files", "settings"]) {
      source.emit(name, JSON.stringify({ name }));
    }
    eq("events.dispatch", received, ["cell", "trace", "job", "jobs", "map", "files", "settings"].map((name) => [name, { name }]));

    const before = received.length;
    source.emit("cell", "{not json");
    eq("events.malformed_ignored", received.length, before);

    source.onerror();
    check("events.error_closes_source", source.closed === true);
    eq("events.backoff_initial_ms", [scheduled.length, scheduled[0].ms], [1, 1000]);

    scheduled[0].fn();
    eq("events.reconnects", eventSources.length, 2);
    const second = eventSources[1];
    second.emit("settings", JSON.stringify({ settings: { theme: "dark" } }));
    eq("events.reconnect_dispatches", received[received.length - 1], ["settings", { settings: { theme: "dark" } }]);

    second.onerror();
    eq("events.backoff_grows", scheduled[1].ms, 2000);
    scheduled[1].fn();
    eventSources[2].onerror();
    eq("events.backoff_next", scheduled[2].ms, 4000);

    stop();
    check("events.stop_closes_source", eventSources[2].closed === true);
  } finally {
    globalThis.setTimeout = realSetTimeout;
  }
});

const SIBLINGS = [
  "pages/cells.js",
  "pages/data.js",
  "components/cell.js",
  "components/markdown.js",
  "components/trace.js",
  "components/plan.js",
  "components/interaction.js",
  "components/progress.js",
  "components/status-bar.js",
  "components/file-tree.js",
];
const gated = SIBLINGS.every((relative) => fs.existsSync(path.join(WEB, relative)));

if (gated) {
  await scenario("shell", async () => {
    const store = await import(webUrl("store.js"));
    store.state.active_workspace = null;
    store.state.map_app_url = null;
    const app = document.createElement("div");
    app.setAttribute("id", "app");
    document.body.append(app);

    const shell = await import(webUrl("pages/shell.js"));
    shell.mountShell(app);
    const firstMenubar = app.querySelector("#menubar");
    const firstMap = app.querySelector("#map-panel");
    check("shell.first_render_regions", Boolean(firstMenubar && firstMap && app.querySelector("#side-panel")));
    check("shell.tab_content", Boolean(app.querySelector("#tab-content")));
    const toggle = app.querySelector("#follow-output");
    check("shell.follow_toggle_in_status_bar", Boolean(toggle && toggle.closest("#status-bar")));

    shell.renderShell();
    check("shell.map_panel_preserved", app.querySelector("#map-panel") === firstMap);
    check("shell.menubar_replaced", app.querySelector("#menubar") !== firstMenubar);

    store.setState({ selected_tab: "Data" });
    check("shell.tab_switch_renders_data", Boolean(app.querySelector("#tab-content")));
    store.setState({ selected_tab: "Cells" });
  });

  await scenario("exports", async () => {
    const expected = {
      "dom.js": ["el", "appendChildren", "toast", "openDialog", "closeDialog", "dialogActions", "formatBytes", "formatTokens", "formatCost", "usageLabel", "usageTitle", "truncate", "prettyValue", "compactPreview", "codeBlock", "expandableContent"],
      "store.js": ["state", "setState", "subscribe", "notify", "isGeneratedCell", "normalizeCells", "jobsFor", "upsertJob", "usageTotals", "applySnapshot"],
      "api.js": ["request", "getState", "updateSettings", "createWorkspace", "openWorkspace", "saveWorkspace", "closeWorkspace", "addCell", "updateCell", "deleteCell", "moveCell", "runCell", "stopCell", "runAll", "respondInteraction", "cancelInteraction", "importFiles", "importUrl", "setMapProject", "reportBridge"],
      "events.js": ["connectEvents"],
      "pages/shell.js": ["mountShell", "renderShell", "renderSidePanel", "renderActiveTab", "renderTabContent"],
      "pages/map.js": ["mountMap", "syncMap"],
      "components/menubar.js": ["renderMenubar", "closeAllMenus"],
      "components/settings-dialog.js": ["openSettingsDialog"],
    };
    for (const [relative, names] of Object.entries(expected)) {
      const module = await import(webUrl(relative));
      const missing = names.filter((name) => typeof module[name] !== "function" && !(name === "state" && module.state));
      eq("exports." + relative.replace(/[/.]/g, "_"), missing, []);
    }

    fetchResponder = () => ({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({
        active_workspace: null,
        workspaces: [],
        cells: [],
        map_project: null,
        map_app_url: null,
        files: [],
        jobs: [],
        settings: { model: "", theme: "light", dangerous_mode: false, max_retries: 5, record_agent_steps: true },
      }),
    });
    await import(webUrl("main.js"));
    await tick();
    const app = document.getElementById("app");
    check("exports.boot_mounts_shell", Boolean(app && app.querySelector("#menubar") && app.querySelector("#side-panel")));
    check(
      "exports.boot_opens_events",
      eventSources.some((source) => !source.closed && source.url === "/api/events")
    );
  });
}

await scenario("map", async () => {
  const store = await import(webUrl("store.js"));
  const map = await import(webUrl("pages/map.js"));
  const ORIGIN = "http://127.0.0.1:8100";
  store.state.active_workspace = "ws1";
  store.state.map_app_url = ORIGIN + "/";
  store.state.map_project = { view: { center: [0, 0] }, layers: [{ id: "a" }] };
  store.state.settings = { ...store.state.settings, theme: "light" };

  const panel = map.mountMap();
  document.body.append(panel);
  const iframes = () => document.querySelectorAll("iframe");
  eq("map.one_iframe", iframes().length, 1);
  const iframe = iframes()[0];
  eq("map.src_has_theme", iframe.getAttribute("src"), ORIGIN + "/index.html?embed=1&theme=light&layout=embed");

  map.syncMap();
  map.syncMap();
  eq("map.iframe_stable", [iframes().length, iframes()[0] === iframe], [1, true]);

  fetchCalls.length = 0;
  sendMessage("http://evil.example", iframe.contentWindow, { type: "geolibre:ready", version: "1.2" });
  sendMessage(ORIGIN, {}, { type: "geolibre:ready", version: "1.2" });
  await tick();
  eq("map.rejects_foreign_origin_and_source", fetchCalls.filter((call) => call.url === "/api/geolibre/bridge").length, 0);

  posted.length = 0;
  fetchCalls.length = 0;
  sendMessage(ORIGIN, iframe.contentWindow, { type: "geolibre:ready", version: "1.2" });
  await tick();
  const handshake = fetchCalls.find((call) => call.url === "/api/geolibre/bridge");
  eq(
    "map.bridge_handshake",
    handshake && handshake.body,
    JSON.stringify({ version: "1.2", methods: ["project.load", "project.request_state"] })
  );
  eq("map.loads_project_on_ready", posted.length, 1);
  eq("map.load_project_message", posted[0].message, { type: "geolibre:load-project", seq: 1, project: store.state.map_project });
  eq("map.post_target_origin", posted[0].target, ORIGIN);

  fetchCalls.length = 0;
  posted.length = 0;
  sendMessage(ORIGIN, iframe.contentWindow, {
    type: "geolibre:state",
    project: { layers: [{ id: "a" }], view: { center: [0, 0] } },
  });
  await tick();
  eq("map.persists_child_state", fetchCalls.filter((call) => call.url === "/api/map/project").length, 1);
  map.syncMap();
  eq("map.suppresses_echo", posted.length, 0);

  posted.length = 0;
  sendMessage(ORIGIN, iframe.contentWindow, { type: "geolibre:state", project: { layers: [{ id: "b" }] } });
  await tick();
  map.syncMap();
  eq("map.reposts_after_divergence", posted.length, 1);
  eq("map.seq_increments", posted[0].message.seq, 2);

  store.state.settings = { ...store.state.settings, theme: "dark" };
  map.syncMap();
  check("map.theme_in_url", iframe.getAttribute("src").includes("theme=dark"), iframe.getAttribute("src"));
  eq("map.theme_keeps_iframe", [iframes().length, iframes()[0] === iframe], [1, true]);

  store.state.active_workspace = "ws2";
  map.syncMap();
  const rebuilt = iframes();
  check("map.rebuilds_on_workspace_switch", rebuilt.length === 1 && rebuilt[0] !== iframe);
  const replacement = rebuilt[0];
  check("map.rebuild_keeps_theme", Boolean(replacement && replacement.getAttribute("src").includes("theme=dark")));

  sendMessage(ORIGIN, replacement.contentWindow, { type: "geolibre:error", message: "boom" });
  const errorNode = document.getElementById("map-error");
  eq("map.error_surface", [errorNode.style.display, errorNode.textContent], ["block", "Map error: boom"]);
});

await scenario("scroll", async () => {
  const scroll = await import(webUrl("scroll.js"));

  // The shell rebuilds the panel; earlier scenarios left theirs behind, and the
  // module finds the live one by id.
  for (const stale of document.querySelectorAll("#tab-content")) stale.remove();

  const panel = document.createElement("div");
  panel.setAttribute("id", "tab-content");
  panel.scrollHeight = 1000;
  panel.clientHeight = 200;
  document.body.append(panel);

  eq("scroll.starts_off", scroll.isFollowing(), false);

  const button = scroll.followToggle();
  const box = button.querySelector("input");
  const label = button.textContent;
  eq("scroll.toggle_starts_off", [scroll.isFollowing(), box.checked, label], [false, false, "Follow output"]);

  // Following is the reader's choice: a scroll while it is off changes nothing.
  panel.scrollTop = 0;
  panel.dispatch("scroll", {});
  eq("scroll.ignores_scroll_while_off", scroll.isFollowing(), false);

  box.checked = true;                        // turn it on: go to the bottom now
  box.dispatch("change", {});
  eq("scroll.toggle_turns_it_on", [scroll.isFollowing(), button.className.includes("active")], [true, true]);
  eq("scroll.jump_goes_to_the_bottom", panel.scrollTop, 1000);

  panel.scrollHeight = 1200;                 // new content arrives
  scroll.hold();
  eq("scroll.follows_new_content", panel.scrollTop, 1200);

  panel.scrollTop = 300;                     // reader scrolled up to read
  panel.dispatch("scroll", {});
  eq("scroll.stops_when_scrolled_away", scroll.isFollowing(), false);
  eq("scroll.toggle_reflects_the_stop", [box.checked, button.className.includes("active")], [false, false]);
  panel.scrollHeight = 1400;
  scroll.hold();
  eq("scroll.keeps_the_readers_place", panel.scrollTop, 300);

  box.checked = true;                        // ticking again takes them back down
  box.dispatch("change", {});
  eq("scroll.toggle_returns_to_the_bottom", [scroll.isFollowing(), panel.scrollTop], [true, 1400]);

  // Our own write is followed by more content before its scroll event lands, so
  // the event reports a position that is no longer the bottom.
  panel.scrollHeight = 2000;
  scroll.hold();
  eq("scroll.write_lands_at_the_bottom", panel.scrollTop, 2000);
  panel.scrollHeight = 2600;
  panel.dispatch("scroll", {});              // the event our own write produced
  eq("scroll.own_write_does_not_stop_following", scroll.isFollowing(), true);
  panel.scrollHeight = 3000;
  scroll.hold();
  eq("scroll.still_following_after_its_own_event", panel.scrollTop, 3000);

  panel.scrollTop = 600;                     // the reader moves again
  panel.dispatch("scroll", {});
  eq("scroll.still_yields_to_the_reader", scroll.isFollowing(), false);

  // The shell swaps the panel on every render; following must survive that.
  box.checked = true;
  box.dispatch("change", {});
  const replacement = document.createElement("div");
  replacement.setAttribute("id", "tab-content");
  replacement.scrollHeight = 4000;
  replacement.clientHeight = 200;
  panel.replaceWith(replacement);
  scroll.scrollTo(replacement, 0);           // a stale position restored by the app
  eq("scroll.own_restore_returns_to_the_bottom", [scroll.isFollowing(), replacement.scrollTop], [true, 4000]);
  replacement.dispatch("scroll", {});
  eq("scroll.own_restore_keeps_following", scroll.isFollowing(), true);

  scroll.unfollow();
  scroll.scrollTo(replacement, 700);         // a restore while reading history
  replacement.dispatch("scroll", {});
  eq(
    "scroll.restore_while_reading_keeps_the_place",
    [replacement.scrollTop, scroll.isFollowing()],
    [700, false]
  );

  const short = document.createElement("div");
  short.scrollHeight = 100;
  short.clientHeight = 200;
  check("scroll.short_content_counts_as_bottom", scroll.atBottom(short) === true);
});

console.log(JSON.stringify({ checks, details, gated }));
"""


@unittest.skipIf(NODE is None, "node is required to exercise the frontend modules")
class FrontendCoreTests(unittest.TestCase):
    report: dict = {}

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "web_core_harness.mjs"
            script.write_text(HARNESS, encoding="utf-8")
            completed = subprocess.run(
                [str(NODE), str(script), str(ROOT)],
                capture_output=True,
                text=True,
                timeout=180,
            )
        if completed.returncode != 0:
            raise AssertionError(
                "node harness failed (%s)\nstdout:\n%s\nstderr:\n%s"
                % (completed.returncode, completed.stdout[-2000:], completed.stderr[-4000:])
            )
        cls.report = json.loads(completed.stdout.strip().splitlines()[-1])

    def assert_group(self, prefix: str) -> None:
        group = {name: ok for name, ok in self.report["checks"].items() if name.startswith(prefix)}
        self.assertTrue(group, f"no {prefix!r} checks were recorded")
        for name, ok in group.items():
            with self.subTest(check=name):
                self.assertTrue(ok, self.report["details"].get(name, ""))

    def test_dom_helpers(self) -> None:
        self.assert_group("dom.")

    def test_store(self) -> None:
        self.assert_group("store.")

    def test_api_endpoints(self) -> None:
        self.assert_group("api.")

    def test_event_stream(self) -> None:
        self.assert_group("events.")

    def test_map_bridge(self) -> None:
        self.assert_group("map.")

    def test_scroll_following(self) -> None:
        self.assert_group("scroll.")

    def test_shell_and_module_exports(self) -> None:
        if not self.report.get("gated"):
            self.skipTest("sibling web modules are not present yet")
        self.assert_group("shell.")
        self.assert_group("exports.")


if __name__ == "__main__":
    unittest.main()
