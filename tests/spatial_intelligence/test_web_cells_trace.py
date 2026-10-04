"""Cells, trace, plan, interaction, progress, and markdown components.

The browser modules are plain ES modules, so each test class materialises a
throwaway ESM project (the real component sources plus stub store/api/dom
siblings), drives it with node, and asserts on the returned DOM tree.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JS = REPO / "spatial_intelligence" / "web" / "js"
# The bundle is downloaded on the first server start (see web/assets.py), so
# these component tests skip on a checkout that has never run the app.
MARKED = REPO / "spatial_intelligence" / "web" / "vendor" / "marked.min.js"
NODE = shutil.which("node")

if not MARKED.is_file():
    MARKED = None

COMPONENTS = ("markdown", "plan", "progress", "trace", "interaction", "cell")

SHIM = r'''
/* Minimal DOM for the component tests: no layout, no CSS engine. */

function kebabCase(name) {
  return String(name).replace(/[A-Z]/g, (ch) => "-" + ch.toLowerCase());
}

class ClassList {
  constructor(node) { this.node = node; }
  values() { return String(this.node.getAttribute("class") || "").split(/\s+/).filter(Boolean); }
  write(values) { this.node.setAttribute("class", values.join(" ")); }
  add(...names) {
    const values = this.values();
    for (const name of names) if (!values.includes(name)) values.push(name);
    this.write(values);
  }
  remove(...names) { this.write(this.values().filter((value) => !names.includes(value))); }
  contains(name) { return this.values().includes(name); }
  toggle(name, force) {
    const want = force === undefined ? !this.contains(name) : Boolean(force);
    if (want) this.add(name); else this.remove(name);
    return want;
  }
}

class ShimText {
  constructor(data) { this.nodeType = 3; this.data = String(data); this.parentNode = null; }
  get textContent() { return this.data; }
  set textContent(value) { this.data = String(value); }
}

class ShimElement {
  constructor(tagName) {
    this.nodeType = 1;
    this.tagName = String(tagName).toUpperCase();
    this.parentNode = null;
    this.attributes = {};
    this.childNodes = [];
    this.style = {};
    this.listeners = {};
    this.checked = false;
    this.disabled = false;
    this.readOnly = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    this._innerHTML = null;
    this._classList = new ClassList(this);
    const attributes = this.attributes;
    this.dataset = new Proxy({}, {
      get: (_, key) => attributes["data-" + kebabCase(key)],
      set: (_, key, value) => { this.setAttribute("data-" + kebabCase(key), value); return true; },
      has: (_, key) => ("data-" + kebabCase(key)) in attributes,
    });
  }
  get value() { return this.attributes.value === undefined ? "" : this.attributes.value; }
  set value(value) { this.setAttribute("value", value == null ? "" : String(value)); }
  get checked() { return this._checked === true; }
  set checked(value) {
    this._checked = Boolean(value);
    if (!this._checked || this.getAttribute("type") !== "radio") return;
    const name = this.getAttribute("name");
    if (!name) return;
    let root = this;
    while (root.parentNode) root = root.parentNode;
    for (const node of descendants(root)) {
      if (node !== this && node.getAttribute("type") === "radio" && node.getAttribute("name") === name) {
        node._checked = false;
      }
    }
  }
  get id() { return this.attributes.id || ""; }
  set id(value) { this.setAttribute("id", value); }
  get className() { return this.attributes.class || ""; }
  set className(value) { this.setAttribute("class", value == null ? "" : String(value)); }
  get classList() { return this._classList; }
  get children() { return this.childNodes.filter((node) => node.nodeType === 1); }
  get firstElementChild() { return this.children[0] || null; }
  get lastElementChild() { return this.children[this.children.length - 1] || null; }
  get textContent() { return this.childNodes.map((node) => node.textContent).join(""); }
  set textContent(value) {
    this._innerHTML = null;
    this.childNodes = [];
    this.append(value == null ? "" : String(value));
  }
  get innerHTML() {
    return this._innerHTML == null ? this.childNodes.map(serialize).join("") : this._innerHTML;
  }
  set innerHTML(value) { this._innerHTML = value == null ? "" : String(value); this.childNodes = []; }
  setAttribute(name, value) {
    if (value === null || value === undefined) return;
    this.attributes[String(name)] = String(value);
  }
  getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
  hasAttribute(name) { return name in this.attributes; }
  removeAttribute(name) { delete this.attributes[name]; }
  append(...nodes) {
    this._innerHTML = null;
    for (const node of nodes) {
      if (node === null || node === undefined) continue;
      const child = typeof node === "string" ? new ShimText(node) : node;
      if (child.parentNode) child.parentNode.removeChild(child);
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }
  appendChild(node) { this.append(node); return node; }
  prepend(...nodes) {
    this._innerHTML = null;
    for (const node of nodes.reverse()) {
      if (node === null || node === undefined) continue;
      const child = typeof node === "string" ? new ShimText(node) : node;
      if (child.parentNode) child.parentNode.removeChild(child);
      child.parentNode = this;
      this.childNodes.unshift(child);
    }
  }
  replaceChildren(...nodes) { this.childNodes = []; this._innerHTML = null; this.append(...nodes); }
  replaceWith(node) {
    const parent = this.parentNode;
    if (!parent) return;
    const index = parent.childNodes.indexOf(this);
    if (index < 0) return;
    if (node.parentNode) node.parentNode.removeChild(node);
    parent.childNodes[index] = node;
    node.parentNode = parent;
    this.parentNode = null;
  }
  removeChild(child) {
    const index = this.childNodes.indexOf(child);
    if (index >= 0) { this.childNodes.splice(index, 1); child.parentNode = null; }
    return child;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(type, listener) { (this.listeners[type] || (this.listeners[type] = [])).push(listener); }
  removeEventListener(type, listener) {
    this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== listener);
  }
  querySelector(selector) { return querySelectorAll(this, selector)[0] || null; }
  querySelectorAll(selector) { return querySelectorAll(this, selector); }
  closest(selector) {
    let node = this;
    while (node && node.nodeType === 1) {
      if (matchesChain(node, tokenize(selector))) return node;
      node = node.parentNode;
    }
    return null;
  }
  scrollIntoView() {}
  focus() {}
  get outerHTML() { return serialize(this); }
}

function serialize(node) {
  if (node.nodeType === 3) return node.data;
  const attrs = Object.entries(node.attributes)
    .map(([name, value]) => " " + name + '="' + value + '"')
    .join("");
  const tag = node.tagName.toLowerCase();
  const inner = node._innerHTML == null ? node.childNodes.map(serialize).join("") : node._innerHTML;
  return "<" + tag + attrs + ">" + inner + "</" + tag + ">";
}

const SIMPLE_RE = /([a-zA-Z][\w-]*)|\.([\w-]+)|#([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/g;

function matchesSimple(node, spec) {
  if (!node || node.nodeType !== 1) return false;
  SIMPLE_RE.lastIndex = 0;
  let matched = false;
  let match;
  while ((match = SIMPLE_RE.exec(spec))) {
    matched = true;
    const tag = match[1];
    const cls = match[2];
    const id = match[3];
    const attr = match[4];
    const attrValue = match[5];
    if (tag && node.tagName.toLowerCase() !== tag.toLowerCase()) return false;
    if (cls && !node.classList.contains(cls)) return false;
    if (id && node.getAttribute("id") !== id) return false;
    if (attr) {
      const value = node.getAttribute(attr);
      if (value == null) return false;
      if (attrValue !== undefined && value !== attrValue) return false;
    }
  }
  return matched;
}

function tokenize(selector) {
  const tokens = String(selector).match(/[^\s>]+|>/g) || [];
  const parts = [];
  let combinator = " ";
  for (const token of tokens) {
    if (token === ">") { combinator = ">"; continue; }
    parts.push({ combinator, spec: token });
    combinator = " ";
  }
  return parts;
}

function ancestors(node) {
  const out = [];
  let parent = node.parentNode;
  while (parent) {
    if (parent.nodeType === 1) out.push(parent);
    parent = parent.parentNode;
  }
  return out;
}

function matchesChain(node, parts, index) {
  if (!node || node.nodeType !== 1) return false;
  if (!matchesSimple(node, parts[index].spec)) return false;
  if (index === 0) return true;
  if (parts[index].combinator === ">") return matchesChain(node.parentNode, parts, index - 1);
  return ancestors(node).some((ancestor) => matchesChain(ancestor, parts, index - 1));
}

function descendants(root) {
  const out = [];
  const stack = [...root.childNodes].reverse();
  while (stack.length) {
    const node = stack.pop();
    if (node.nodeType !== 1) continue;
    out.push(node);
    for (const child of [...node.childNodes].reverse()) stack.push(child);
  }
  return out;
}

function querySelectorAll(root, selector) {
  const parts = tokenize(selector);
  if (!parts.length) return [];
  return descendants(root).filter((node) => matchesChain(node, parts, parts.length - 1));
}

const documentElement = new ShimElement("html");
const head = new ShimElement("head");
const body = new ShimElement("body");
documentElement.append(head, body);

globalThis.document = {
  documentElement,
  head,
  body,
  createElement: (tag) => new ShimElement(tag),
  createTextNode: (text) => new ShimText(text),
  getElementById: (id) => descendants(documentElement).find((node) => node.getAttribute("id") === id) || null,
  querySelector: (selector) => querySelectorAll(documentElement, selector)[0] || null,
  querySelectorAll: (selector) => querySelectorAll(documentElement, selector),
  addEventListener() {},
  removeEventListener() {},
};

globalThis.window = {
  setTimeout,
  clearTimeout,
  requestAnimationFrame: (callback) => setTimeout(callback, 0),
};

export function collect(root) { return descendants(root); }

export function dump(node) {
  if (node === null || node === undefined) return null;
  if (node.nodeType === 3) return { text: node.data };
  const entry = {
    tag: node.tagName.toLowerCase(),
    class: node.getAttribute("class") || "",
    attrs: { ...node.attributes },
    text: node.textContent,
    html: node.innerHTML,
    style: { ...node.style },
    children: node.childNodes.map(dump),
  };
  if (node.checked) entry.checked = true;
  if (node.disabled) entry.disabled = true;
  if (node.readOnly) entry.readOnly = true;
  if (node.value) entry.value = node.value;
  return entry;
}

export function fire(node, type, event = {}) {
  const payload = { type, target: node, preventDefault() {}, stopPropagation() {}, ...event };
  for (const listener of (node.listeners && node.listeners[type]) || []) listener(payload);
  return payload;
}

export function tick(ms = 10) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
'''

DOM_STUB = r'''
/* Test stand-in for web/js/dom.js: the same node helpers the app exports. */

export function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2), value);
      } else if (value !== null && value !== undefined) {
        node.setAttribute(key, value);
      }
    }
  }
  if (children != null) {
    for (const child of Array.isArray(children) ? children : [children]) {
      if (child == null) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
  }
  return node;
}

export function appendChildren(node, ...children) {
  node.append(...children.filter((child) => child != null));
}

export function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  if (bytes < 1024) return bytes + " B";
  const units = ["KB", "MB", "GB", "TB"];
  let amount = bytes;
  let unit = -1;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return amount.toFixed(amount >= 10 || unit === 0 ? 0 : 1) + " " + units[unit];
}

export function formatTokens(n) {
  n = Number(n) || 0;
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

export function formatCost(c) {
  if (c == null) return null;
  const n = Number(c);
  if (!isFinite(n) || n === 0) return null;
  return "$" + (n < 0.001 ? n.toExponential(1) : n.toFixed(4));
}

export function usageLabel(u) {
  const parts = [];
  if (u.input_tokens != null) parts.push("\u2191" + formatTokens(u.input_tokens));
  if (u.output_tokens != null) parts.push("\u2193" + formatTokens(u.output_tokens));
  const cost = formatCost(u.cost);
  if (cost) parts.push(cost);
  return parts.join(" ");
}

export function usageTitle(u) {
  const parts = [];
  if (u.requests != null) parts.push(u.requests + " request" + (u.requests === 1 ? "" : "s"));
  if (u.tool_calls != null) parts.push(u.tool_calls + " tool call" + (u.tool_calls === 1 ? "" : "s"));
  if (u.total_tokens != null) parts.push(u.total_tokens + " total tokens");
  if (u.cache_read_tokens) parts.push(u.cache_read_tokens + " cached read tokens");
  return parts.join(" \u00b7 ");
}

export function truncate(s, n) {
  s = String(s == null ? "" : s);
  return s.length > n ? s.slice(0, n) + "\u2026" : s;
}

export function prettyValue(value) {
  if (typeof value === "string") {
    const t = value.trim();
    if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
      try {
        return JSON.stringify(JSON.parse(value), null, 2);
      } catch (_) {
        /* not valid JSON */
      }
    }
    return value;
  }
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch (_) {
    return String(value);
  }
}

export function compactPreview(value) {
  let s;
  if (typeof value === "string") s = value;
  else if (value == null) s = "";
  else {
    try {
      s = JSON.stringify(value);
    } catch (_) {
      s = String(value);
    }
  }
  return truncate(String(s).replace(/\s+/g, " ").trim(), 100);
}

export function codeBlock(content) {
  const pre = el("pre", { class: "trace-code" });
  pre.append(el("code", { text: content == null ? "" : String(content) }));
  return pre;
}

export function expandableContent(label, content, extraClass = "") {
  const text = String(content == null ? "" : content);
  const details = el("details", {
    class: "agent-action-content" + (extraClass ? " " + extraClass : ""),
  });
  const summary = el("summary", {});
  summary.append(
    el("strong", { text: label }),
    el("span", { class: "agent-action-preview", text: compactPreview(text) })
  );
  details.append(summary, el("pre", { text }));
  return details;
}

export function toast() {}

export function openDialog() {}

export function closeDialog() {}

export function dialogActions() {}
'''

STORE_STUB = r'''
/* Test stand-in for web/js/store.js (frozen interface). */

export const state = {
  active_workspace: "demo",
  workspaces: [],
  cells: [],
  jobs: [],
  files: [],
  settings: { model: "", theme: "light", dangerous_mode: false, max_retries: 5, record_agent_steps: true },
  selected_tab: "Cells",
};

const listeners = new Set();

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function notify() {
  for (const listener of listeners) listener(state);
}

export function setState(patch) {
  Object.assign(state, patch);
  notify();
  return state;
}

export function isGeneratedCell(cell) {
  const geoai = (cell && cell.metadata && cell.metadata.geoai) || {};
  return Boolean(geoai.generated || geoai.kind === "interaction");
}

export function normalizeCells(cells) {
  return Array.isArray(cells)
    ? cells
        .filter((cell) => cell && typeof cell === "object" && !isGeneratedCell(cell))
        .map((cell) => {
          const normalized = { ...cell };
          if (Object.prototype.hasOwnProperty.call(cell, "source")) {
            normalized.source = cell.source == null ? "" : String(cell.source);
          }
          return normalized;
        })
    : [];
}

export function jobsFor(cellId) {
  return (state.jobs || []).filter((job) => job.parent_id === cellId);
}

/** Mirrors the real store's step identity (see web/js/store.js). */
export function stepKey(step) {
  if (!step) return "";
  if (step.tool_call_id) return step.type + ":" + step.tool_call_id;
  if (step.type === "plan") return "plan:" + JSON.stringify(step.items || []);
  if (step.type === "usage") return "usage:" + JSON.stringify(step.usage || null);
  return (
    step.type + ":" + (step.name || "") + ":" + String(step.content == null ? "" : step.content)
  );
}

export function upsertJob(job) {
  const jobs = state.jobs || [];
  const index = jobs.findIndex((item) => item.job_id === job.job_id);
  if (index === -1) state.jobs = [...jobs, job];
  else state.jobs[index] = { ...jobs[index], ...job };
  return job;
}

export function usageTotals() {
  let input = 0;
  let output = 0;
  let requests = 0;
  let cost = 0;
  let has = false;
  for (const cell of state.cells || []) {
    const usage = cell.usage;
    if (!usage) continue;
    has = true;
    input += usage.input_tokens || 0;
    output += usage.output_tokens || 0;
    requests += usage.requests || 0;
    cost += Number(usage.cost) || 0;
  }
  return { input, output, requests, cost, cacheRead: 0, has };
}
'''

API_STUB = r'''
/* Test stand-in for web/js/api.js (frozen interface). */

import { state } from "./store.js";

export async function request() {}

export async function getState() {
  return { cells: state.cells, jobs: state.jobs };
}

export async function addCell(kind) {
  const cell = {
    id: "new-" + kind,
    kind,
    source: "",
    outputs: [],
    execution_count: null,
    status: "idle",
    metadata: { geoai: {} },
    trace: [],
    usage: null,
    interaction: null,
    interaction_history: [],
  };
  state.cells = [...state.cells, cell];
  return { cells: state.cells, jobs: state.jobs };
}

export async function updateCell() {}

export async function deleteCell() {}

export async function moveCell() {}

export async function runCell() {}

export async function stopCell() {}

export async function runAll() {}

export async function respondInteraction() {
  return { status: "running" };
}

export async function cancelInteraction() {
  return { status: "stopped" };
}
'''

HARNESS = r'''
import "./shim.mjs";
import { createRequire } from "node:module";
import { collect, dump, fire, tick } from "./shim.mjs";
import { state, subscribe, upsertJob } from "./js/store.js";
import { el } from "./js/dom.js";
import * as markdown from "./js/components/markdown.js";
import * as plan from "./js/components/plan.js";
import * as progress from "./js/components/progress.js";
import * as trace from "./js/components/trace.js";
import * as interaction from "./js/components/interaction.js";
import * as cellModule from "./js/components/cell.js";
import * as cellsPage from "./js/pages/cells.js";

const require = createRequire(import.meta.url);
const marked = require("./vendor/marked.min.js");
globalThis.marked = marked;

function makeCell(overrides) {
  return Object.assign({
    id: "c1",
    kind: "python",
    source: "print(1)",
    outputs: [],
    execution_count: 1,
    status: "idle",
    metadata: { geoai: {} },
    trace: [],
    usage: null,
    interaction: null,
    interaction_history: [],
  }, overrides);
}

function makeJob(overrides) {
  return Object.assign({
    job_id: "j1",
    kind: "download",
    status: "running",
    label: "data.zip",
    unit: "bytes",
    completed: 0,
    total: null,
    detail: null,
    artifact: null,
    error: null,
    parent_id: null,
  }, overrides);
}

function markedDump() {
  return { renderer: typeof marked.Renderer, parse: typeof marked.parse };
}

const scenarios = {
  exports() {
    const names = (module, list) => list.map((name) => typeof module[name]);
    return {
      markdown: names(markdown, ["renderMarkdown", "visibleTraceGroups", "escapeHtml"]),
      plan: names(plan, ["renderPlan", "planFromTrace"]),
      progress: names(progress, ["renderJob", "updateJob"]),
      trace: names(trace, ["renderTrace", "groupTraceSteps", "renderStepNode", "repaintTrace"]),
      interaction: names(interaction, ["renderInteraction", "revealInteraction"]),
      cell: names(cellModule, ["renderCell"]),
      cells: names(cellsPage, ["renderCellsTab", "renderCellsOnly", "renderAddCellRow", "editMarkdown", "focusCell", "addCell"]),
      aliases: {
        renderJobNode: progress.renderJobNode === progress.renderJob,
        updateJobNode: progress.updateJobNode === progress.updateJob,
        applyTrace: typeof trace.applyTrace,
        flushPendingTrace: typeof trace.flushPendingTrace,
        reconcile: typeof interaction.reconcilePendingInteraction,
      },
    };
  },

  async markdown() {
    globalThis.marked = marked;
    const table = markdown.renderMarkdown("| a | b |\n| - | - |\n| 1 | 2 |");
    const escaped = markdown.renderMarkdown("<script>alert('x')</script>\n\nok");
    delete globalThis.marked;
    let missing = null;
    try {
      markdown.renderMarkdown("x");
    } catch (error) {
      missing = error.message;
    }
    globalThis.marked = marked;
    const groups = [
      { type: "usage", usage: { input_tokens: 1 } },
      { type: "text", content: "thinking" },
      { type: "tool", call: { name: "run_python" }, result: null },
    ];
    return {
      marked_module: markedDump(),
      table_has_table: table.includes("<table"),
      table_has_cell: table.includes("<td>1</td>"),
      escaped_has_script: escaped.includes("<script"),
      escaped_has_lt: escaped.includes("&lt;script&gt;"),
      missing_message: missing,
      escaped_value: markdown.escapeHtml('a"b<c>&d'),
      done_groups: markdown.visibleTraceGroups(groups, "done", true).map((group) => group.type),
      running_groups: markdown.visibleTraceGroups(groups, "running", true).map((group) => group.type),
      done_without_output: markdown.visibleTraceGroups(groups, "done", false).map((group) => group.type),
      output_text: markdown.cellOutputText({
        outputs: [
          { output_type: "stream", text: "hello\n" },
          { output_type: "error", ename: "ValueError", evalue: "boom", traceback: ["line1", "line2"] },
          { output_type: "display_data" },
        ],
      }),
      empty_output_text: markdown.cellOutputText({}),
    };
  },

  async progress() {
    const indeterminate = progress.renderJob(makeJob({ job_id: "j1", parent_id: "c1", detail: "Fetching tiles", completed: 10 }));
    const known = progress.renderJob(makeJob({ job_id: "j2", completed: 100, total: 200 }));
    const done = progress.renderJob(makeJob({ job_id: "j3", status: "done", completed: 200, total: 200, artifact: "data/out.zip" }));
    const failed = progress.renderJob(makeJob({ job_id: "j4", status: "error", error: "boom", completed: 5 }));
    const inTrace = progress.renderJob(makeJob({ job_id: "j5", parent_id: "c1" }), "trace");
    const live = progress.renderJob(makeJob({ job_id: "j6" }));
    const updated = progress.updateJob(live, makeJob({ job_id: "j6", status: "done", completed: 4, total: 4 }));
    const skill = progress.renderJob(makeJob({
      job_id: "j7",
      kind: "skill",
      status: "done",
      label: "Loaded skill torchgeo/api/models.md",
      unit: "lines",
      completed: 124,
      total: 124,
      detail: "api/models.md (repo)",
      parent_id: "c1",
    }));
    return {
      indeterminate: dump(indeterminate),
      known: dump(known),
      done: dump(done),
      failed: dump(failed),
      in_trace: dump(inTrace),
      updated: dump(updated),
      skill: dump(skill),
      updated_in_place: updated === live,
    };
  },

  async grouping() {
    const grouped = trace.groupTraceSteps([
      { type: "text", content: "one" },
      { type: "text_delta", content: " two" },
      { type: "tool_call", name: "run_python", args: { code: "1+1" }, tool_call_id: "t1" },
      { type: "tool_result", name: "run_python", content: "2", tool_call_id: "t1" },
      { type: "tool_call", name: "write_file", args: { path: "a.txt" }, tool_call_id: "t2" },
      { type: "tool_result", name: "write_file", content: "ok", tool_call_id: "t2" },
      { type: "usage", usage: { input_tokens: 5 } },
    ]);
    const byName = trace.groupTraceSteps([
      { type: "tool_call", name: "only_name", args: {} },
      { type: "tool_result", name: "only_name", content: "done" },
    ]);
    const orphan = trace.groupTraceSteps([{ type: "tool_result", name: "lost", content: "x" }]);
    const summaries = grouped.map((group) => ({
      type: group.type,
      name: (group.call && group.call.name) || (group.result && group.result.name) || null,
      paired: Boolean(group.result),
      content: group.content || null,
    }));
    return {
      summaries,
      by_name: byName.map((group) => ({ type: group.type, paired: Boolean(group.result) })),
      orphan: orphan.map((group) => ({ type: group.type, call: group.call, name: group.result && group.result.name })),
      tool: dump(trace.renderStepNode(grouped[1])),
      text: dump(trace.renderStepNode(grouped[0])),
      usage: trace.renderStepNode(grouped[3]),
      empty: trace.renderStepNode(null),
    };
  },

  async layout() {
    globalThis.marked = marked;
    // A finished download step with a message before it and one after it.
    const steps = [
      { type: "text_delta", content: "Looking for the file." },
      { type: "tool_call", name: "download", args: { url: "http://x/f.geojson" }, tool_call_id: "t1" },
      { type: "tool_result", name: "download", content: "data/f.geojson", tool_call_id: "t1" },
      { type: "text_delta", content: "Saved it." },
    ];
    const cell = makeCell({ id: "c1", kind: "prompt", source: "download it", status: "running", trace: steps });
    const card = (anchor) =>
      makeJob({
        job_id: "j" + String(anchor),
        parent_id: "c1",
        status: "done",
        label: "f.geojson",
        unit: "bytes",
        completed: 3145728,
        total: 3145728,
        anchor,
      });
    const outline = (node) =>
      [...node.children].map((child) => ({
        class: child.getAttribute("class"),
        job: child.getAttribute("data-job-id"),
        source: child.dataset.source || null,
        text: child.textContent.replace(/\s+/g, " ").trim().slice(0, 60),
      }));

    // (a) A job opened after the two steps of the download call renders there:
    //     behind the tool call, ahead of the message that followed it.
    const anchored = outline(cellModule.renderCell(cell, [card(2)]).querySelector(".trace"));
    // (b) A job whose server reported no anchor can only go last.
    const legacy = outline(
      cellModule.renderCell(cell, [makeJob({ job_id: "j9", parent_id: "c1", status: "done" })]).querySelector(".trace")
    );

    // (c) Streamed live, the same cell and job must draw in that same order —
    //     including after a repaint, which is what a tab switch or a workspace
    //     refresh does.
    state.cells = [{ ...JSON.parse(JSON.stringify(cell)), trace: [] }];
    state.jobs = [];
    state.selected_tab = "Cells";
    const content = el("div", { id: "tab-content" });
    document.body.append(content);
    cellsPage.renderCellsOnly();
    for (const step of steps) trace.applyTrace({ id: "c1", step });
    const streamed = outline(document.querySelector('.cell[data-cell-id="c1"] .trace'));
    upsertJob(card(2));
    trace.repaintTrace(state.cells[0]);
    const repainted = outline(document.querySelector('.cell[data-cell-id="c1"] .trace'));
    trace.repaintTrace(state.cells[0]);
    const twice = outline(document.querySelector('.cell[data-cell-id="c1"] .trace'));

    // (d) A tool result that lands after a repaint fills the call it answers
    //     instead of splitting into a second node.
    const live = state.cells[0];
    live.trace = [];
    trace.repaintTrace(live);
    trace.applyTrace({ id: "c1", step: { type: "text_delta", content: "Reading files." } });
    trace.applyTrace({ id: "c1", step: { type: "tool_call", name: "list_files", args: {}, tool_call_id: "t2" } });
    const midCall = document.querySelector('.cell[data-cell-id="c1"] .trace details[data-trace-key]');
    trace.repaintTrace(live); // an unrelated repaint lands mid-call
    trace.applyTrace({ id: "c1", step: { type: "tool_result", name: "list_files", content: "a.tif", tool_call_id: "t2" } });
    const paired = dump(document.querySelector('.cell[data-cell-id="c1"] .trace'));

    // (e) A tool call the reader expanded stays open across a repaint.
    const container = document.querySelector('.cell[data-cell-id="c1"] .trace');
    const details = container.querySelector("details[data-trace-key]");
    details.setAttribute("open", "open");
    trace.repaintTrace(live);
    const reopened = document
      .querySelector('.cell[data-cell-id="c1"] .trace')
      .querySelector("details[data-trace-key]")
      .hasAttribute("open");

    return {
      anchored,
      legacy,
      streamed,
      repainted,
      twice,
      mid_call: midCall && midCall.getAttribute("class"),
      paired,
      reopened,
    };
  },

  async apply_trace() {
    state.cells = [makeCell({ id: "c9", kind: "prompt", trace: [] })];
    const box = el("div", { class: "cell", "data-cell-id": "c9" });
    const container = el("div", { class: "trace" });
    box.append(container);
    document.body.append(box);
    trace.applyTrace({ id: "c9", step: { type: "text_delta", content: "hi" } });
    trace.applyTrace({ id: "c9", step: { type: "text_delta", content: "!" } });
    trace.applyTrace({ id: "unknown", step: { type: "text", content: "ignored" } });

    // Steps that arrive before the snapshot introducing their cell are held,
    // then replayed once it lands: the snapshot already carries the tool call.
    trace.applyTrace({
      id: "boot",
      step: { type: "tool_call", name: "list_files", args: {}, tool_call_id: "t1" },
    });
    trace.applyTrace({
      id: "boot",
      step: { type: "tool_result", name: "list_files", content: "a.tif", tool_call_id: "t1" },
    });
    trace.applyTrace({ id: "boot", step: { type: "text_delta", content: "done" } });
    state.cells.push(
      makeCell({
        id: "boot",
        kind: "prompt",
        status: "running",
        trace: [{ type: "tool_call", name: "list_files", args: {}, tool_call_id: "t1" }],
      })
    );
    const bootBox = el("div", { class: "cell", "data-cell-id": "boot" });
    const bootTrace = el("div", { class: "trace" });
    bootBox.append(bootTrace);
    document.body.append(bootBox);
    trace.flushPendingTrace();

    // Once the snapshot has landed the buffer is closed: a step for a cell the
    // snapshot never carried is dropped instead of growing the buffer forever.
    trace.applyTrace({ id: "gone", step: { type: "text", content: "late" } });

    return {
      trace_length: state.cells[0].trace.length,
      container_count: container.childNodes.length,
      text_source: container.lastElementChild.dataset.source,
      text_html: container.lastElementChild.innerHTML,
      cell_count: state.cells.length,
      boot_steps: state.cells[1].trace.map((step) => step.type),
      boot_nodes: bootTrace.childNodes.map((node) => node.getAttribute("class")),
      late_step_stored: state.cells.some((entry) =>
        (entry.trace || []).some((step) => step.content === "late")
      ),
    };
  },

  async live_trace() {
    globalThis.marked = marked;
    // A prompt cell that has just started running: empty trace, no jobs yet.
    // Its trace container must exist anyway, because streamed steps are
    // appended by cell id into the live DOM rather than by re-rendering.
    state.cells = [
      makeCell({
        id: "live",
        kind: "prompt",
        source: "list them",
        status: "running",
        trace: [],
      }),
    ];
    document.body.append(cellModule.renderCell(state.cells[0], []));
    const mounted = document.querySelector('.cell[data-cell-id="live"] .trace');
    const mounted_empty = mounted !== null && mounted.childNodes.length === 0;
    const elements = (node) => (node ? node.childNodes.filter((child) => child.nodeType === 1) : []);

    trace.applyTrace({
      id: "live",
      step: { type: "tool_call", name: "list_files", args: {}, tool_call_id: "t1" },
    });
    trace.applyTrace({
      id: "live",
      step: { type: "tool_result", name: "list_files", content: "a.tif", tool_call_id: "t1" },
    });
    trace.applyTrace({
      id: "live",
      step: { type: "plan", items: [{ id: "s1", content: "List files", status: "completed" }] },
    });

    const container = document.querySelector('.cell[data-cell-id="live"] .trace');
    const nodes = elements(container);
    const call = nodes.find((node) => node.classList.contains("tool-call")) || null;
    return {
      mounted_before_steps: mounted !== null,
      mounted_empty,
      same_container: mounted === container,
      element_count: nodes.length,
      call_pending: call ? call.classList.contains("pending") : null,
      has_plan: nodes.some((node) => node.classList.contains("trace-plan")),
      trace_length: state.cells[0].trace.length,
      text: container ? container.textContent : null,
    };
  },

  async render_trace() {
    globalThis.marked = marked;
    const done = makeCell({
      id: "c1",
      kind: "prompt",
      status: "done",
      execution_count: 3,
      outputs: [{ output_type: "stream", text: "final answer" }],
      trace: [
        { type: "text", content: "thinking" },
        { type: "tool_call", name: "run_python", args: { code: "1+1" }, tool_call_id: "t1" },
        { type: "tool_result", name: "run_python", content: "2", tool_call_id: "t1" },
        { type: "text", content: "final answer" },
      ],
    });
    const waiting = makeCell({
      id: "c2",
      kind: "prompt",
      status: "waiting_for_input",
      trace: [{ type: "tool_call", name: "request_user_input", args: {}, tool_call_id: "q1" }],
      interaction: {
        id: "i1",
        title: "Pick a region",
        prompt: "Where?",
        tool_call_id: "q1",
        fields: [{ id: "f1", label: "Region", type: "radio", options: [{ value: "a", label: "A" }] }],
      },
    });
    const planCell = makeCell({
      id: "c3",
      kind: "prompt",
      status: "running",
      trace: [
        {
          type: "plan",
          items: [
            { id: "1", content: "load data", status: "completed" },
            { id: "2", content: "clip", status: "in_progress", active_form: "clipping" },
          ],
        },
      ],
    });
    return {
      done: trace.renderTrace(done).map(dump),
      done_count: trace.renderTrace(done).length,
      waiting: trace.renderTrace(waiting).map(dump),
      plan: trace.renderTrace(planCell).map(dump),
    };
  },

  async interaction() {
    const request = {
      id: "i1",
      title: "Input required",
      prompt: "Configure the run",
      fields: [
        {
          id: "region",
          label: "Region",
          type: "radio",
          options: [
            { value: "north", label: "North", recommended: true, description: "cold", thumbnail_url: "http://x/t.png" },
            { value: "south", label: "South" },
          ],
        },
        {
          id: "layers",
          label: "Layers",
          type: "multi_select",
          options: [{ value: "a", label: "A" }, { value: "b", label: "B" }],
        },
        { id: "note", label: "Note", type: "text", placeholder: "why", default: "draft" },
        { id: "confirm", label: "Confirm", type: "confirmation", default: true },
      ],
    };
    const cell = makeCell({ id: "c1", kind: "prompt", status: "waiting_for_input", interaction: request });
    const rendered = dump(interaction.renderInteraction(cell));

    const submitting = interaction.renderInteraction(cell);
    const inputs = collect(submitting).filter((node) => node.tagName === "INPUT");
    const radios = inputs.filter((node) => node.getAttribute("type") === "radio");
    const checkboxes = inputs.filter((node) => node.getAttribute("type") === "checkbox" && String(node.getAttribute("name")).includes("layers"));
    radios[1].checked = true;
    inputs.find((node) => node.getAttribute("type") === "text").value = "because";
    checkboxes[0].checked = true;
    fire(submitting, "submit");
    await tick(20);
    const submitted = {
      status: cell.status,
      interaction: cell.interaction,
      history_length: cell.interaction_history.length,
      answers: cell.interaction_history[0] && cell.interaction_history[0].answers,
    };

    const emptyRequest = {
      id: "i2",
      title: "Input required",
      prompt: "",
      fields: [{ id: "note", label: "Note", type: "text" }],
    };
    const invalidCell = makeCell({ id: "c2", kind: "prompt", status: "waiting_for_input", interaction: emptyRequest });
    const invalidForm = interaction.renderInteraction(invalidCell);
    fire(invalidForm, "submit");
    await tick(20);
    const invalid = {
      error_text: (invalidForm.querySelector(".interaction-error") || {}).textContent,
      interaction: invalidCell.interaction,
      history_length: invalidCell.interaction_history.length,
    };

    const cancelCell = makeCell({ id: "c3", kind: "prompt", status: "waiting_for_input", interaction: request });
    const cancelForm = interaction.renderInteraction(cancelCell);
    const cancelButton = collect(cancelForm).find(
      (node) => node.tagName === "BUTTON" && node.textContent === "Cancel"
    );
    fire(cancelButton, "click");
    await tick(20);
    const cancelled = { status: cancelCell.status, interaction: cancelCell.interaction };

    const submittedView = dump(interaction.renderInteraction(cell, {
      ...request,
      submitted: true,
      answers: { region: "north", layers: ["b"], note: "kept", confirm: false },
    }));

    const revealCell = makeCell({ id: "c4", kind: "prompt", status: "waiting_for_input", interaction: request });
    const revealBox = el("div", { class: "cell", "data-cell-id": "c4" });
    revealBox.append(interaction.renderInteraction(revealCell));
    document.body.append(revealBox);
    interaction.revealInteraction("c4");
    await tick(20);
    interaction.revealInteraction("missing");
    await tick(20);

    state.cells = [makeCell({
      id: "c7",
      kind: "prompt",
      status: "waiting_for_input",
      interaction: { id: "i7", title: "T", prompt: "P", fields: [{ id: "f", label: "L", type: "text" }] },
    })];
    state.selected_tab = "Cells";
    const content = el("div", { id: "tab-content" });
    document.body.append(content);
    await interaction.reconcilePendingInteraction("c7");
    await tick(20);
    const reconciled = {
      repainted: content.childNodes.length,
      has_cell: Boolean(content.querySelector('.cell[data-cell-id="c7"]')),
      has_form: Boolean(content.querySelector(".interaction-form")),
    };

    return { rendered, submitted, invalid, cancelled, submitted_view: submittedView, reconciled };
  },

  async cell() {
    globalThis.marked = marked;
    const pythonJob = makeJob({ job_id: "j1", parent_id: "p1", label: "tiles.zip" });
    const python = dump(cellModule.renderCell(
      makeCell({ id: "p1", kind: "python", source: "print(1)", execution_count: 2, status: "done", outputs: [{ output_type: "stream", text: "1\n" }] }),
      [pythonJob]
    ));
    const plain = dump(cellModule.renderCell(makeCell({ id: "p2", kind: "python", source: "x" }), []));
    const tool = dump(cellModule.renderCell(
      makeCell({
        id: "t1",
        kind: "tool",
        source: "",
        status: "done",
        execution_count: 5,
        metadata: { geoai: { tool_name: "run_python", args: { code: "print(1)" } } },
        outputs: [{ output_type: "stream", text: "1\n" }],
      }),
      [makeJob({ job_id: "j2", parent_id: "t1", status: "done", completed: 1, total: 1, unit: "steps" })]
    ));
    const prompt = dump(cellModule.renderCell(
      makeCell({
        id: "r1",
        kind: "prompt",
        source: "why?",
        status: "done",
        execution_count: 4,
        usage: { input_tokens: 1200, output_tokens: 30, cost: 0.0002, requests: 2 },
        outputs: [{ output_type: "stream", text: "answer" }],
        trace: [{ type: "text", content: "thinking" }, { type: "text", content: "answer" }],
      }),
      []
    ));
    const markdownCell = dump(cellModule.renderCell(
      makeCell({ id: "m1", kind: "markdown", source: "# Title\n\nbody", metadata: { geoai: {} } }),
      [makeJob({ job_id: "j3", parent_id: "m1", status: "running" })]
    ));
    const running = dump(cellModule.renderCell(
      makeCell({ id: "r2", kind: "prompt", source: "go", status: "running" }),
      []
    ));
    const waiting = dump(cellModule.renderCell(
      makeCell({ id: "r3", kind: "python", source: "input()", status: "waiting_for_input", execution_count: 7 }),
      []
    ));
    const generated = dump(cellModule.renderCell(
      makeCell({ id: "g1", kind: "python", source: "gen", metadata: { geoai: { generated: true } } }),
      []
    ));
    const failed = dump(cellModule.renderCell(
      makeCell({ id: "e1", kind: "python", source: "boom", status: "error", outputs: [{ output_type: "error", ename: "ValueError", evalue: "bad" }] }),
      []
    ));
    return { python, plain, tool, prompt, markdown: markdownCell, running, waiting, generated, failed };
  },

  async cells_page() {
    globalThis.marked = marked;
    state.active_workspace = "demo";
    state.selected_tab = "Cells";
    state.cells = [
      makeCell({ id: "p1", kind: "python", source: "1" }),
      makeCell({ id: "t1", kind: "tool", source: "", metadata: { geoai: { tool_name: "download" } } }),
    ];
    state.jobs = [
      makeJob({ job_id: "j1", parent_id: "p1", label: "tiles.zip" }),
      makeJob({ job_id: "j2", parent_id: "t1", status: "done", completed: 1, total: 1, unit: "steps" }),
      makeJob({ job_id: "j3", parent_id: "gone", label: "orphan.csv" }),
    ];
    const tab = dump(cellsPage.renderCellsTab());

    const content = el("div", { id: "tab-content" });
    document.body.append(content);
    cellsPage.renderCellsOnly();
    const repainted = content.querySelectorAll(".cell").length;

    state.active_workspace = null;
    const noWorkspace = dump(cellsPage.renderCellsTab());
    state.active_workspace = "demo";
    state.cells = [];
    state.jobs = [];
    const noCells = dump(cellsPage.renderCellsTab());

    const row = cellsPage.renderAddCellRow();
    const buttons = collect(row).filter((node) => node.tagName === "BUTTON").map((node) => node.textContent);

    state.cells = [makeCell({ id: "m1", kind: "markdown", source: "# Old", metadata: { geoai: {} } })];
    state.jobs = [];
    cellsPage.renderCellsOnly();
    cellsPage.editMarkdown(state.cells[0]);
    const editing = dump(content);
    const textarea = content.querySelector("textarea");
    textarea.value = "# New";
    fire(textarea, "blur");
    await tick(20);
    const afterEdit = dump(content);
    const editedSource = state.cells[0].source;

    const unsubscribe = subscribe(() => cellsPage.renderCellsOnly());
    const before = state.cells.length;
    const addButton = collect(row).find((node) => node.textContent === "+ Python");
    fire(addButton, "click");
    await tick(30);
    const added = {
      before,
      after: state.cells.length,
      kind: state.cells[state.cells.length - 1].kind,
      repainted: Boolean(content.querySelector('.cell[data-cell-id="new-python"]')),
    };
    unsubscribe();

    return { tab, repainted, no_workspace: noWorkspace, no_cells: noCells, buttons, editing, after_edit: afterEdit, edited_source: editedSource, added };
  },
};

const name = process.argv[2];
const runner = scenarios[name];
if (!runner) {
  console.error("unknown scenario: " + name);
  process.exit(2);
}
process.stdout.write(JSON.stringify(await runner()));
'''


def walk(node):
    """Yield every element node of a dumped tree, depth first."""
    if not isinstance(node, dict) or "tag" not in node:
        return
    yield node
    for child in node.get("children", ()):
        yield from walk(child)


def find_all(node, tag=None, cls=None, attr=None, value=None, text=None):
    """Return every dumped element matching the given selector-ish filters."""
    found = []
    for item in walk(node):
        if tag is not None and item["tag"] != tag:
            continue
        if cls is not None and cls not in item.get("class", "").split():
            continue
        if attr is not None and item.get("attrs", {}).get(attr) != value:
            continue
        if text is not None and text not in item.get("text", ""):
            continue
        found.append(item)
    return found


class ScenarioCase(unittest.TestCase):
    """Materialise a throwaway ESM project and run one harness scenario."""

    SCENARIO = ""

    @classmethod
    def setUpClass(cls):
        if NODE is None:
            raise unittest.SkipTest("node is not available")
        if MARKED is None:
            raise unittest.SkipTest("the marked vendor bundle is not available")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        components = cls.root / "js" / "components"
        pages = cls.root / "js" / "pages"
        vendor = cls.root / "vendor"
        for directory in (components, pages, vendor):
            directory.mkdir(parents=True)
        for name in COMPONENTS:
            shutil.copy2(JS / "components" / f"{name}.js", components / f"{name}.js")
        shutil.copy2(JS / "pages" / "cells.js", pages / "cells.js")
        shutil.copy2(JS / "scroll.js", cls.root / "js" / "scroll.js")
        shutil.copy2(MARKED, vendor / "marked.min.js")
        (cls.root / "js" / "dom.js").write_text(DOM_STUB, encoding="utf-8")
        (cls.root / "js" / "store.js").write_text(STORE_STUB, encoding="utf-8")
        (cls.root / "js" / "api.js").write_text(API_STUB, encoding="utf-8")
        (cls.root / "shim.mjs").write_text(SHIM, encoding="utf-8")
        (cls.root / "harness.mjs").write_text(HARNESS, encoding="utf-8")
        cls.result = cls.run_node(cls.SCENARIO)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def run_node(cls, scenario):
        completed = subprocess.run(
            [NODE, "harness.mjs", scenario],
            cwd=cls.root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"node failed for scenario {scenario!r}:\n{completed.stdout}\n{completed.stderr}"
            )
        return json.loads(completed.stdout)

    def one(self, node, **kwargs):
        found = find_all(node, **kwargs)
        self.assertEqual(len(found), 1, f"expected one {kwargs}, found {len(found)}")
        return found[0]


class ExportsTests(ScenarioCase):
    SCENARIO = "exports"

    def test_every_contract_export_is_a_function(self):
        groups = ["markdown", "plan", "progress", "trace", "interaction", "cell", "cells"]
        for group in groups:
            with self.subTest(group=group):
                self.assertEqual(self.result[group], ["function"] * len(self.result[group]))

    def test_shell_aliases_resolve_to_the_same_implementations(self):
        self.assertEqual(
            self.result["aliases"],
            {
                "renderJobNode": True,
                "updateJobNode": True,
                "applyTrace": "function",
                "flushPendingTrace": "function",
                "reconcile": "function",
            },
        )


class MarkdownTests(ScenarioCase):
    SCENARIO = "markdown"

    def test_marked_is_used_at_call_time_and_renders_gfm(self):
        self.assertTrue(self.result["table_has_table"])
        self.assertTrue(self.result["table_has_cell"])

    def test_raw_html_is_escaped(self):
        self.assertFalse(self.result["escaped_has_script"])
        self.assertTrue(self.result["escaped_has_lt"])
        self.assertEqual(self.result["escaped_value"], "a&quot;b&lt;c&gt;&amp;d")

    def test_missing_renderer_raises_a_clear_error(self):
        self.assertEqual(self.result["missing_message"], "The local marked renderer was not loaded")

    def test_visible_trace_groups_hide_usage_and_final_answer(self):
        self.assertEqual(self.result["done_groups"], ["tool"])
        self.assertEqual(self.result["running_groups"], ["text", "tool"])
        self.assertEqual(self.result["done_without_output"], ["text", "tool"])

    def test_cell_output_text_joins_streams_and_errors(self):
        self.assertEqual(self.result["output_text"], "hello\n\nValueError: boom\nline1\nline2")
        self.assertEqual(self.result["empty_output_text"], "")


class ProgressTests(ScenarioCase):
    SCENARIO = "progress"

    def test_indeterminate_job_has_no_percentage(self):
        node = self.result["indeterminate"]
        self.assertEqual(node["attrs"]["data-job-id"], "j1")
        progress = self.one(node, cls="download-progress")
        self.assertIn("indeterminate", progress["class"])
        self.assertEqual(self.one(progress, cls="download-progress-fill")["style"]["width"], "35%")
        self.assertEqual(self.one(node, cls="download-progress-label")["text"], "10 B · running")
        self.assertEqual(self.one(node, cls="download-description")["text"], "Fetching tiles")

    def test_known_total_renders_a_percentage(self):
        node = self.result["known"]
        progress = self.one(node, cls="download-progress")
        self.assertNotIn("indeterminate", progress["class"])
        self.assertEqual(self.one(progress, cls="download-progress-fill")["style"]["width"], "50%")
        self.assertEqual(self.one(node, cls="download-progress-label")["text"], "50% · 100 B / 200 B")

    def test_completed_job_shows_its_artifact(self):
        node = self.result["done"]
        self.assertEqual(self.one(node, cls="download-progress-fill")["style"]["width"], "100%")
        self.assertEqual(self.one(node, cls="download-progress-label")["text"], "100% · 200 B")
        self.assertEqual(self.one(node, cls="download-description")["text"], "Saved to data/out.zip")
        status = self.one(node, cls="download-status")
        self.assertEqual(status["text"], "complete")
        self.assertIn("status-done", status["class"])

    def test_failed_job_shows_the_error(self):
        node = self.result["failed"]
        self.assertEqual(self.one(node, cls="download-error")["text"], "boom")
        self.assertEqual(self.one(node, cls="download-status")["text"], "failed")
        self.assertEqual(self.one(node, cls="download-progress-fill")["style"]["width"], "0")

    def test_a_skill_card_names_the_page_it_read(self):
        node = self.result["skill"]
        self.assertEqual(self.one(node, cls="badge")["text"], "skill")
        self.assertEqual(self.one(node, cls="download-filename")["text"], "Loaded skill torchgeo/api/models.md")
        self.assertEqual(self.one(node, cls="download-description")["text"], "api/models.md (repo)")
        self.assertEqual(self.one(node, cls="download-progress-label")["text"], "100% · 124 lines")
        self.assertEqual(self.one(node, cls="download-status")["text"], "complete")

    def test_standalone_and_cell_strip_variants(self):
        standalone = self.result["indeterminate"]
        self.assertIn("download-cell", standalone["class"])
        self.assertIn("cell", standalone["class"])
        stripped = self.result["in_trace"]
        self.assertIn("download-trace", stripped["class"])
        self.assertNotIn("cell", stripped["class"].split())
        self.assertEqual(self.one(stripped, cls="download-trace-summary")["tag"], "div")

    def test_update_job_mutates_the_existing_node(self):
        self.assertTrue(self.result["updated_in_place"])
        node = self.result["updated"]
        self.assertEqual(node["attrs"]["data-job-id"], "j6")
        self.assertEqual(self.one(node, cls="download-progress-fill")["style"]["width"], "100%")
        self.assertEqual(self.one(node, cls="download-status")["text"], "complete")


class TraceGroupingTests(ScenarioCase):
    SCENARIO = "grouping"

    def test_text_deltas_merge_and_tool_pairs_join_by_id(self):
        self.assertEqual(
            self.result["summaries"],
            [
                {"type": "text", "name": None, "paired": False, "content": "one two"},
                {"type": "tool", "name": "run_python", "paired": True, "content": None},
                {"type": "tool", "name": "write_file", "paired": True, "content": None},
                {"type": "usage", "name": None, "paired": False, "content": None},
            ],
        )

    def test_tool_pairs_join_by_name_when_the_id_is_missing(self):
        self.assertEqual(self.result["by_name"], [{"type": "tool", "paired": True}])

    def test_unmatched_result_becomes_its_own_group(self):
        self.assertEqual(self.result["orphan"], [{"type": "tool", "call": None, "name": "lost"}])

    def test_step_nodes_render_per_group_type(self):
        tool = self.result["tool"]
        self.assertIn("tool-call", tool["class"])
        self.assertIn("run_python", tool["text"])
        text = self.result["text"]
        self.assertIn("trace-text", text["class"])
        self.assertEqual(text["attrs"]["data-source"], "one two")
        self.assertIn("<p>one two</p>", text["html"])
        self.assertIsNone(self.result["usage"])
        self.assertIsNone(self.result["empty"])


class TraceLayoutTests(ScenarioCase):
    """Where a progress card sits in the trace, live and after a repaint.

    Regression: the renderer drew every step first and every job card last, so
    any repaint — switching tabs, refreshing the workspace from the Data tab, the
    end of a run — moved a card past the messages streamed after it, and a tool
    result arriving after a repaint could no longer fill its pending call.
    """

    SCENARIO = "layout"

    def test_a_card_renders_at_the_step_that_opened_it(self):
        classes = [child["class"] for child in self.result["anchored"]]
        self.assertEqual(
            classes,
            [
                "trace-step trace-text markdown",
                "trace-step tool-call",
                "trace-step download-trace download-progress-node",
                "trace-step trace-text markdown",
            ],
        )
        texts = [child["source"] for child in self.result["anchored"]]
        self.assertEqual(texts, ["Looking for the file.", None, None, "Saved it."])

    def test_a_card_without_an_anchor_goes_last(self):
        classes = [child["class"] for child in self.result["legacy"]]
        self.assertEqual(classes[-1], "trace-step download-trace download-progress-node")

    def test_the_live_stream_then_its_card_match_a_full_render(self):
        self.assertEqual(
            [child["class"] for child in self.result["streamed"]],
            [
                "trace-step trace-text markdown",
                "trace-step tool-call",
                "trace-step trace-text markdown",
            ],
        )
        self.assertEqual(self.result["repainted"], self.result["anchored"])
        self.assertEqual(self.result["twice"], self.result["anchored"])

    def test_a_result_arriving_after_a_repaint_fills_its_call(self):
        self.assertIn("pending", self.result["mid_call"])
        nodes = self.result["paired"]["children"]
        self.assertEqual([node["class"] for node in nodes].count("trace-step tool-call"), 1)
        self.assertEqual(find_all(self.result["paired"], cls="tool-result"), [])
        call = find_all(self.result["paired"], cls="tool-call")[0]
        self.assertNotIn("pending", call["class"])
        self.assertEqual(
            [label["text"] for label in find_all(call, cls="trace-io-label")], ["Input", "Output"]
        )
        self.assertIn("a.tif", call["text"])

    def test_a_repaint_keeps_a_tool_call_the_reader_opened_open(self):
        self.assertTrue(self.result["reopened"])


class ApplyTraceTests(ScenarioCase):
    """Streamed steps landing on their cell, including the boot window.

    Regression: the event stream opens before the first snapshot is applied, so
    steps published in between used to be dropped on the floor — a browser
    reload mid-run rebuilt only part of the trace.
    """

    SCENARIO = "apply_trace"

    def test_streamed_steps_are_stored_and_appended(self):
        self.assertEqual(self.result["trace_length"], 2)
        self.assertEqual(self.result["container_count"], 1)
        self.assertEqual(self.result["text_source"], "hi!")
        self.assertIn("<p>hi!</p>", self.result["text_html"])

    def test_steps_buffered_before_the_snapshot_are_replayed_once(self):
        self.assertEqual(
            self.result["boot_steps"], ["tool_call", "tool_result", "text_delta"]
        )
        # The replayed result fills the call the snapshot already carried, and
        # the text follows it: the buffer draws the same nodes a reload would.
        self.assertEqual(
            self.result["boot_nodes"],
            ["trace-step tool-call", "trace-step trace-text markdown"],
        )

    def test_the_buffer_closes_once_the_snapshot_has_landed(self):
        self.assertEqual(self.result["cell_count"], 2)
        self.assertFalse(self.result["late_step_stored"])


class LiveTraceMountTests(ScenarioCase):
    """A running cell must have somewhere for streamed steps to land.

    Regression: the trace container used to be mounted only when the cell
    already had trace nodes or jobs, so every live step was dropped by
    ``applyTrace`` and the whole run appeared at once when it finished.
    """

    SCENARIO = "live_trace"

    def test_a_running_prompt_cell_mounts_an_empty_trace_container(self):
        self.assertTrue(
            self.result["mounted_before_steps"],
            "no .trace container on a running prompt cell",
        )
        self.assertTrue(self.result["mounted_empty"])

    def test_streamed_steps_land_in_that_same_container(self):
        self.assertTrue(self.result["same_container"])
        self.assertGreaterEqual(self.result["element_count"], 2)
        self.assertEqual(self.result["trace_length"], 3)
        self.assertIn("list_files", self.result["text"])
        self.assertIn("a.tif", self.result["text"])

    def test_a_tool_result_resolves_its_pending_call_and_the_plan_renders(self):
        self.assertIs(self.result["call_pending"], False)
        self.assertTrue(self.result["has_plan"])


class RenderTraceTests(ScenarioCase):
    SCENARIO = "render_trace"

    def test_done_cell_hides_the_final_answer_and_keeps_the_tool_call(self):
        nodes = self.result["done"]
        self.assertEqual(len(nodes), 2)
        self.assertIn("trace-text", nodes[0]["class"])
        self.assertEqual(nodes[0]["attrs"]["data-source"], "thinking")
        self.assertIn("tool-call", nodes[1]["class"])
        self.assertNotIn("pending", nodes[1]["class"])
        self.assertIn("1+1", nodes[1]["text"])
        self.assertEqual(self.result["done_count"], 2)

    def test_waiting_cell_marks_the_call_and_mounts_the_interaction(self):
        waiting = self.result["waiting"]
        self.assertEqual(len(waiting), 2)
        self.assertIn("waiting", waiting[0]["class"])
        self.assertIn("input required", waiting[0]["text"])
        self.assertEqual(waiting[1]["class"], "interaction-form")
        self.assertIn("Pick a region", waiting[1]["text"])

    def test_plan_step_renders_a_plan_node(self):
        plan = self.result["plan"]
        self.assertEqual(len(plan), 1)
        self.assertIn("trace-plan", plan[0]["class"])
        self.assertIn("clipping", plan[0]["text"])
        self.assertIn("1/2 done", plan[0]["text"])


class InteractionTests(ScenarioCase):
    SCENARIO = "interaction"

    def test_all_field_types_render_with_thumbnails_and_defaults(self):
        form = self.result["rendered"]
        self.assertEqual(form["tag"], "form")
        self.assertEqual(form["class"], "interaction-form")
        self.assertIn("Configure the run", form["text"])
        self.assertEqual(len(find_all(form, cls="interaction-field")), 4)
        thumbnails = find_all(form, tag="img")
        self.assertEqual([img["attrs"]["src"] for img in thumbnails], ["http://x/t.png"])
        radios = find_all(form, tag="input", attr="type", value="radio")
        self.assertEqual([bool(radio.get("checked")) for radio in radios], [True, False])
        self.assertIn("Recommended", form["text"])
        confirmations = find_all(form, tag="input", attr="type", value="checkbox")
        self.assertTrue(confirmations[-1]["checked"])
        self.assertEqual(self.one(form, tag="input", attr="type", value="text")["attrs"]["value"], "draft")

    def test_submit_posts_answers_and_clears_the_pending_interaction(self):
        submitted = self.result["submitted"]
        self.assertEqual(submitted["status"], "running")
        self.assertIsNone(submitted["interaction"])
        self.assertEqual(submitted["history_length"], 1)
        self.assertEqual(
            submitted["answers"],
            {"region": "south", "layers": ["a"], "note": "because", "confirm": True},
        )

    def test_missing_required_answer_blocks_submission(self):
        invalid = self.result["invalid"]
        self.assertEqual(invalid["error_text"], "Please complete Note.")
        self.assertIsNotNone(invalid["interaction"])
        self.assertEqual(invalid["history_length"], 0)

    def test_cancel_stops_the_cell(self):
        self.assertEqual(self.result["cancelled"], {"status": "stopped", "interaction": None})

    def test_submitted_view_is_read_only(self):
        view = self.result["submitted_view"]
        self.assertEqual(view["tag"], "div")
        self.assertEqual(view["class"], "interaction-form submitted")
        self.assertIn("Selection submitted", view["text"])
        self.assertEqual(find_all(view, tag="button"), [])
        text = self.one(view, tag="input", attr="type", value="text")
        self.assertTrue(text["disabled"])
        self.assertEqual(text["attrs"]["value"], "kept")
        radios = find_all(view, tag="input", attr="type", value="radio")
        self.assertEqual([bool(radio.get("checked")) for radio in radios], [True, False])

    def test_reconcile_repaints_the_cells_tab(self):
        reconciled = self.result["reconciled"]
        self.assertEqual(reconciled["repainted"], 1)
        self.assertTrue(reconciled["has_cell"])
        self.assertTrue(reconciled["has_form"])


class CellTests(ScenarioCase):
    SCENARIO = "cell"

    def test_python_cell_mounts_the_job_strip(self):
        cell = self.result["python"]
        self.assertEqual(cell["attrs"]["data-cell-id"], "p1")
        strip = self.one(cell, cls="trace")
        job = self.one(strip, cls="download-trace")
        self.assertEqual(job["attrs"]["data-job-id"], "j1")
        self.assertIn("tiles.zip", job["text"])
        self.assertTrue(self.one(cell, cls="cell-out")["text"].startswith("Out["))
        self.assertIn("1", self.one(cell, cls="markdown")["html"])
        self.assertEqual(len(find_all(cell, tag="textarea")), 1)

    def test_cell_without_jobs_has_no_job_strip(self):
        self.assertEqual(find_all(self.result["plain"], cls="trace"), [])

    def test_tool_cell_mounts_its_own_job_strip(self):
        cell = self.result["tool"]
        self.assertIn("agent tool", cell["text"])
        self.assertEqual(self.one(cell, cls="agent-action-name")["text"], "run_python")
        self.assertEqual(find_all(cell, tag="textarea"), [])
        self.assertEqual(find_all(cell, cls="cell-out"), [])
        job = self.one(cell, cls="download-trace")
        self.assertEqual(job["attrs"]["data-job-id"], "j2")
        self.assertEqual(self.one(job, cls="download-status")["text"], "complete")

    def test_prompt_cell_shows_usage_trace_and_hides_the_final_text(self):
        cell = self.result["prompt"]
        usage = self.one(cell, cls="usage")
        self.assertEqual(usage["text"], "\u21911.2k \u219330 $2.0e-4")
        self.assertEqual(usage["attrs"]["title"], "2 requests")
        texts = find_all(cell, cls="trace-text")
        self.assertEqual([node["attrs"]["data-source"] for node in texts], ["thinking"])

    def test_markdown_cell_renders_body_and_keeps_its_job_strip(self):
        cell = self.result["markdown"]
        self.assertEqual(find_all(cell, cls="cell-out"), [])
        self.assertEqual(find_all(cell, tag="textarea"), [])
        self.assertIn("<h1>Title</h1>", self.one(cell, cls="markdown")["html"])
        self.assertEqual(self.one(cell, cls="download-trace")["attrs"]["data-job-id"], "j3")
        labels = [button["attrs"]["title"] for button in find_all(cell, tag="button")]
        self.assertEqual(labels, ["Edit", "Delete"])

    def test_running_and_waiting_statuses_render_their_own_controls(self):
        running = self.result["running"]
        self.assertIn("stop", self.one(running, tag="button", attr="title", value="Stop")["class"])
        self.assertIn("running", self.one(running, cls="cell-out")["text"])
        waiting = self.result["waiting"]
        self.assertIn("waiting for input", self.one(waiting, cls="cell-out")["text"])
        self.assertEqual(find_all(waiting, cls="run-btn"), [])
        self.assertEqual(self.one(self.one(waiting, cls="cell-header"), cls="counter")["text"], "In[7]")

    def test_generated_cells_are_read_only(self):
        cell = self.result["generated"]
        self.assertIn("generated-cell", cell["class"])
        self.assertTrue(self.one(cell, tag="textarea")["readOnly"])

    def test_error_output_renders_as_preformatted_error(self):
        cell = self.result["failed"]
        self.assertEqual(self.one(cell, tag="pre", cls="error")["text"], "ValueError: bad")


class CellsPageTests(ScenarioCase):
    SCENARIO = "cells_page"

    def test_cells_and_orphan_jobs_are_listed_separately(self):
        cells = self.result["tab"]["children"]
        self.assertEqual(
            [child["attrs"].get("data-cell-id") for child in cells if "cell" in child["class"].split()],
            ["p1", "t1", None],
        )
        orphan = cells[-1]
        self.assertIn("download-cell", orphan["class"])
        self.assertNotIn("data-cell-id", orphan["attrs"])
        self.assertEqual(orphan["attrs"]["data-job-id"], "j3")
        python = next(child for child in cells if child["attrs"].get("data-cell-id") == "p1")
        self.assertEqual(self.one(python, cls="download-trace")["attrs"]["data-job-id"], "j1")

    def test_render_cells_only_repaints_the_tab(self):
        self.assertEqual(self.result["repainted"], 3)

    def test_empty_states(self):
        self.assertIn("No workspace open", self.result["no_workspace"]["children"][0]["text"])
        self.assertIn("No cells yet", self.result["no_cells"]["children"][0]["text"])

    def test_add_row_offers_every_cell_kind(self):
        self.assertEqual(self.result["buttons"], ["+ Markdown", "+ Python", "+ Prompt"])

    def test_edit_markdown_swaps_in_a_textarea_and_saves_on_blur(self):
        editing = self.result["editing"]
        self.assertEqual([node["tag"] for node in find_all(editing, tag="textarea")], ["textarea"])
        self.assertEqual(self.one(editing, tag="textarea")["value"], "# Old")
        self.assertEqual(self.result["edited_source"], "# New")
        self.assertIn("<h1>New</h1>", self.one(self.result["after_edit"], cls="markdown")["html"])

    def test_add_cell_appends_and_repaints(self):
        added = self.result["added"]
        self.assertEqual(added["after"], added["before"] + 1)
        self.assertEqual(added["kind"], "python")
        self.assertTrue(added["repainted"])


if __name__ == "__main__":
    unittest.main()
