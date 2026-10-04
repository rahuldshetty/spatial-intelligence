"""Web modules under ``spatial_intelligence/web/js``: markdown, trace, progress, Data tab.

Every case loads the real ES modules in Node.js through an absolute ``file:///``
import and inspects the values/DOM trees they produce. No network, no bundler.
"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[2] / "spatial_intelligence" / "web"
WEB_JS = WEB_ROOT / "js"
MARKED_JS = WEB_ROOT / "vendor" / "marked.min.js"
NODE = shutil.which("node")


def module_url(*parts) -> str:
    path = WEB_JS.joinpath(*parts)
    return path.as_uri()


DOM_SHIM = r"""
/*
 * Minimal DOM used by the frontend modules in Node.js: element/text nodes with
 * class, attribute, style and event bookkeeping plus simple CSS-selector
 * lookups (tag, .class, #id, [attr]). Layout and rendering are not modelled.
 */
function ShimNode(tag, nodeType) {
  this.nodeType = nodeType === undefined ? 1 : nodeType;
  this.tagName = String(tag).toUpperCase();
  this.children = [];
  this.attributes = {};
  this.listeners = {};
  this.style = {};
  this.dataset = {};
  this.className = "";
  this.parentElement = null;
  this._text = "";
  this._value = "";
}

Object.defineProperty(ShimNode.prototype, "textContent", {
  get() {
    if (this.children.length) return this.children.map((c) => c.textContent).join("");
    return this._text;
  },
  set(value) {
    this.children = [];
    this._text = String(value);
  },
});

Object.defineProperty(ShimNode.prototype, "id", {
  get() {
    return this.attributes.id || "";
  },
  set(value) {
    this.attributes.id = String(value);
  },
});

Object.defineProperty(ShimNode.prototype, "value", {
  get() {
    return this._value;
  },
  set(value) {
    this._value = String(value);
  },
});

Object.defineProperty(ShimNode.prototype, "files", {
  get() {
    return this._value === "" ? [] : String(this._value).split(",");
  },
});

Object.defineProperty(ShimNode.prototype, "classList", {
  get() {
    const node = this;
    return {
      add(...names) {
        node._setClasses([...node._classes(), ...names]);
      },
      remove(...names) {
        node._setClasses(node._classes().filter((name) => !names.includes(name)));
      },
      contains(name) {
        return node._classes().includes(name);
      },
      toggle(name, force) {
        const has = node._classes().includes(name);
        const next = force === undefined ? !has : Boolean(force);
        if (next) node.classList.add(name);
        else node.classList.remove(name);
        return next;
      },
    };
  },
});

ShimNode.prototype._classes = function () {
  return String(this.className || "")
    .split(/\s+/)
    .filter(Boolean);
};

ShimNode.prototype._setClasses = function (names) {
  this.className = [...new Set(names.filter(Boolean))].join(" ");
  this.attributes.class = this.className;
};

ShimNode.prototype.append = function (...nodes) {
  for (const node of nodes) {
    if (node === null || node === undefined) continue;
    const child = node instanceof ShimNode ? node : document.createTextNode(String(node));
    child.parentElement = this;
    this.children.push(child);
  }
  return this;
};

ShimNode.prototype.appendChild = function (node) {
  return this.append(node);
};

ShimNode.prototype.prepend = function (...nodes) {
  for (const node of nodes.reverse()) {
    if (node === null || node === undefined) continue;
    const child = node instanceof ShimNode ? node : document.createTextNode(String(node));
    child.parentElement = this;
    this.children.unshift(child);
  }
  return this;
};

ShimNode.prototype.replaceChildren = function (...nodes) {
  this.children = [];
  this.append(...nodes);
  return this;
};

ShimNode.prototype.replaceWith = function (node) {
  const parent = this.parentElement;
  if (!parent) return;
  const index = parent.children.indexOf(this);
  if (index === -1) return;
  node.parentElement = parent;
  parent.children[index] = node;
  this.parentElement = null;
};

ShimNode.prototype.remove = function () {
  const parent = this.parentElement;
  if (!parent) return;
  const index = parent.children.indexOf(this);
  if (index !== -1) parent.children.splice(index, 1);
  this.parentElement = null;
};

ShimNode.prototype.setAttribute = function (name, value) {
  this.attributes[name] = String(value);
  if (name === "class") this.className = String(value);
};

ShimNode.prototype.getAttribute = function (name) {
  return Object.prototype.hasOwnProperty.call(this.attributes, name)
    ? this.attributes[name]
    : null;
};

ShimNode.prototype.addEventListener = function (type, handler) {
  this.listeners[type] = this.listeners[type] || [];
  this.listeners[type].push(handler);
};

ShimNode.prototype.click = function () {
  for (const handler of this.listeners.click || []) handler({ target: this });
};

ShimNode.prototype.focus = function () {};
ShimNode.prototype.scrollIntoView = function () {};
ShimNode.prototype.closest = function () {
  return null;
};
ShimNode.prototype.getBoundingClientRect = function () {
  return { top: 0, left: 0, width: 0, height: 0 };
};

ShimNode.prototype.contains = function (node) {
  if (node === this) return true;
  return this.children.some((child) => child.contains(node));
};

ShimNode.prototype.descendants = function () {
  const out = [];
  for (const child of this.children) out.push(child, ...child.descendants());
  return out;
};

ShimNode.prototype.querySelectorAll = function (selector) {
  const parts = selector.trim().split(/\s+/);
  const ancestorParts = parts.slice(0, -1);
  return this.descendants().filter(
    (node) =>
      matchesSelector(node, parts[parts.length - 1]) &&
      hasAncestorChain(node, ancestorParts, this)
  );
};

function hasAncestorChain(node, parts, root) {
  let index = parts.length - 1;
  if (index < 0) return true;
  let current = node.parentElement;
  while (current && current !== root) {
    if (index >= 0 && matchesSelector(current, parts[index])) index -= 1;
    if (index < 0) return true;
    current = current.parentElement;
  }
  return index < 0;
}

ShimNode.prototype.querySelector = function (selector) {
  return this.querySelectorAll(selector)[0] || null;
};

function matchesSelector(node, selector) {
  for (const attr of selector.matchAll(/\[([^\]=]+)(?:=("([^"]*)"|'([^']*)'|[^\]]+))?\]/g)) {
    if (!Object.prototype.hasOwnProperty.call(node.attributes, attr[1])) return false;
    if (attr[2] !== undefined) {
      const expected = attr[3] !== undefined ? attr[3] : attr[4] !== undefined ? attr[4] : attr[2];
      if (String(node.attributes[attr[1]]) !== expected) return false;
    }
  }
  const base = selector.replace(/\[[^\]]*\]/g, "").trim();
  if (!base) return true;
  const id = base.match(/#([^.#]+)/);
  if (id && node.attributes.id !== id[1]) return false;
  const tag = base.match(/^([a-zA-Z][a-zA-Z0-9-]*)/);
  if (tag && node.tagName !== tag[1].toUpperCase()) return false;
  for (const cls of base.matchAll(/\.([^.#]+)/g)) {
    if (!node._classes().includes(cls[1])) return false;
  }
  return true;
}

globalThis.ShimNode = ShimNode;
globalThis.document = {
  body: new ShimNode("body"),
  documentElement: new ShimNode("html"),
  createElement: (tag) => new ShimNode(tag),
  createTextNode: (text) => {
    const node = new ShimNode("#text", 3);
    node._text = String(text);
    return node;
  },
  getElementById: (id) => document.body.querySelector("#" + id),
  querySelector: (selector) => document.body.querySelector(selector),
  querySelectorAll: (selector) => document.body.querySelectorAll(selector),
  addEventListener: () => {},
  removeEventListener: () => {},
};
globalThis.window = globalThis;
globalThis.location = { href: "http://localhost/", origin: "http://localhost" };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.EventSource = class {
  constructor() {}
  addEventListener() {}
  close() {}
};
/* Never settles: modules may kick off a state load on import, and a rejection
   would both hit the network and crash Node through an unhandled rejection. */
globalThis.fetch = () => new Promise(() => {});
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.requestAnimationFrame = (callback) => setTimeout(callback, 0);
globalThis.cancelAnimationFrame = (handle) => clearTimeout(handle);
globalThis.matchMedia = () => ({
  matches: false,
  addEventListener() {},
  removeEventListener() {},
});
"""


@unittest.skipUnless(NODE, "Node.js is required to load the frontend ES modules")
class WebRenderingTestCase(unittest.TestCase):
    """Runs a snippet in Node against a real module and returns its JSON result."""

    def run_module(self, module: str, body: str, *, marked: bool = False) -> object:
        script = ["import { createRequire } from 'node:module';", DOM_SHIM]
        if marked:
            script.append(
                "const require = createRequire(" + json.dumps(str(MARKED_JS)) + ");"
            )
            script.append(
                "const markedExport = require(" + json.dumps(str(MARKED_JS)) + ");"
            )
            script.append(
                "globalThis.marked = typeof markedExport.parse === 'function'"
                " ? markedExport : markedExport.marked;"
            )
        script.append("const mod = await import(" + json.dumps(module) + ");")
        script.append("const out = await (async () => {" + body + "})();")
        script.append("console.log(JSON.stringify(out));")
        result = subprocess.run(
            [NODE, "--input-type=module", "-e", "\n".join(script)],
            cwd=str(WEB_ROOT.parent.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        if result.returncode != 0:
            self.fail("node failed for " + module + ":\n" + result.stderr.strip())
        return json.loads(result.stdout.strip().splitlines()[-1])


@unittest.skipUnless(MARKED_JS.exists(), "The downloaded Marked asset is required")
class MarkdownRenderingTests(WebRenderingTestCase):
    def test_markdown_renders_gfm_table_and_inline_formatting(self):
        source = (
            "| Layer | Cloud |\n"
            "|---|---:|\n"
            "| **Before** | `0%` |\n"
            "| **After** | 24% |"
        )

        html = self.run_module(
            module_url("components", "markdown.js"),
            "return mod.renderMarkdown(" + json.dumps(source) + ");",
            marked=True,
        )

        self.assertIn("<table>", html)
        self.assertIn("<strong>Before</strong>", html)
        self.assertIn("<code>0%</code>", html)
        self.assertIn('align="right"', html)

    def test_markdown_escapes_raw_html(self):
        html = self.run_module(
            module_url("components", "markdown.js"),
            "return mod.renderMarkdown("
            + json.dumps("<img src=x onerror=alert(1)>")
            + ");",
            marked=True,
        )

        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)


class TraceVisibilityTests(WebRenderingTestCase):
    def test_usage_and_final_text_hide_once_the_cell_is_done_with_output(self):
        groups = [
            {"type": "tool", "call": {"name": "list_files"}},
            {"type": "text", "content": "Done."},
            {"type": "usage", "usage": {"total_tokens": 10}},
        ]

        visible = self.run_module(
            module_url("components", "markdown.js"),
            "return mod.visibleTraceGroups("
            + json.dumps(groups)
            + ", 'done', true).map((group) => group.type);",
        )

        self.assertEqual(visible, ["tool"])

    def test_running_text_stays_visible_and_usage_never_renders(self):
        groups = [
            {"type": "text", "content": "Working"},
            {"type": "usage", "usage": {"total_tokens": 10}},
        ]

        visible = self.run_module(
            module_url("components", "markdown.js"),
            "return mod.visibleTraceGroups("
            + json.dumps(groups)
            + ", 'running', false).map((group) => group.type);",
        )

        self.assertEqual(visible, ["text"])

    def test_rendered_trace_shows_tool_step_without_the_usage_step(self):
        cell = {
            "id": "cell-1",
            "status": "done",
            "outputs": [{"output_type": "stream", "text": "ok"}],
            "trace": [
                {
                    "type": "tool_call",
                    "name": "list_files",
                    "tool_call_id": "t1",
                    "args": {"pattern": "*.tif"},
                },
                {
                    "type": "tool_result",
                    "name": "list_files",
                    "tool_call_id": "t1",
                    "content": "roads.tif",
                },
                {"type": "usage", "usage": {"total_tokens": 10, "cost": 0.5}},
            ],
        }

        rendered = self.run_module(
            module_url("components", "trace.js"),
            "const container = document.createElement('div');"
            "container.append(...mod.renderTrace(" + json.dumps(cell) + "));"
            "const steps = container.querySelectorAll('.trace-step');"
            "return {"
            "  text: container.textContent,"
            "  steps: steps.length,"
            "  pending: steps.filter((step) => step.classList.contains('pending')).length,"
            "  usageSteps: container.querySelectorAll('.trace-step.usage').length,"
            "  toolSteps: container.querySelectorAll('.trace-step.tool-call').length,"
            "};",
        )

        self.assertEqual(rendered["toolSteps"], 1)
        self.assertEqual(rendered["steps"], 1)
        self.assertEqual(rendered["pending"], 0)
        self.assertEqual(rendered["usageSteps"], 0)
        self.assertIn("list_files", rendered["text"])
        self.assertIn("roads.tif", rendered["text"])


class ProgressRenderingTests(WebRenderingTestCase):
    def render_job(self, job: dict) -> dict:
        return self.run_module(
            module_url("components", "progress.js"),
            "const node = mod.renderJob(" + json.dumps(job) + ");"
            "const progress = node.querySelector('.download-progress');"
            "const fill = node.querySelector('.download-progress-fill');"
            "return {"
            "  indeterminate: progress.classList.contains('indeterminate'),"
            "  width: fill.style.width || null,"
            "  label: node.querySelector('.download-progress-label').textContent,"
            "  text: node.textContent,"
            "};",
        )

    def test_unknown_total_renders_an_indeterminate_job(self):
        rendered = self.render_job(
            {
                "job_id": "job-1",
                "status": "running",
                "kind": "download",
                "label": "roads.tif",
                "completed": 0,
                "total": None,
                "unit": "bytes",
            }
        )

        self.assertTrue(rendered["indeterminate"])
        self.assertIn("roads.tif", rendered["text"])
        self.assertIn("running", rendered["label"])
        self.assertNotIn("%", rendered["label"])

    def test_known_total_renders_a_percentage(self):
        rendered = self.render_job(
            {
                "job_id": "job-2",
                "status": "running",
                "kind": "download",
                "label": "roads.tif",
                "completed": 512,
                "total": 1024,
                "unit": "bytes",
            }
        )

        self.assertFalse(rendered["indeterminate"])
        self.assertEqual(rendered["width"], "50%")
        self.assertTrue(rendered["label"].startswith("50%"))
        self.assertIn("512 B", rendered["label"])


class FileTreeRenderingTests(WebRenderingTestCase):
    def test_directories_sort_first_and_nest_with_open_details(self):
        rendered = self.run_module(
            module_url("components", "file-tree.js"),
            "const tree = mod.renderFileTree(['zones/b.tif', 'zones/a.tif', 'roads.geojson']);"
            "return {"
            "  classes: tree.className,"
            "  top: [...tree.children].map((li) => li.className),"
            "  dirOpen: tree.querySelector('.tree-dir details').attributes.open === '',"
            "  dirChildren: [...tree.querySelector('.tree-dir li')"
            "    .parentElement.querySelectorAll('.tree-file .tree-name')]"
            "    .map((span) => span.textContent),"
            "  topNames: [...tree.querySelectorAll('.tree-file .tree-name')]"
            "    .map((span) => span.textContent),"
            "};",
        )

        self.assertEqual(rendered["classes"], "file-tree")
        self.assertEqual(rendered["top"], ["tree-dir", "tree-file"])
        self.assertTrue(rendered["dirOpen"])
        self.assertEqual(rendered["dirChildren"], ["a.tif", "b.tif"])
        self.assertIn("roads.geojson", rendered["topNames"])

    def test_build_file_tree_groups_paths_under_their_directories(self):
        rendered = self.run_module(
            module_url("components", "file-tree.js"),
            "const root = mod.buildFileTree(['a/b/c.tif', 'a/d.tif']);"
            "const a = root.children.get('a');"
            "return {"
            "  rootIsDir: root.isDir,"
            "  aIsDir: a.isDir,"
            "  aChildren: [...a.children.keys()].sort(),"
            "  nestedIsDir: a.children.get('b').isDir,"
            "  leafIsDir: a.children.get('b').children.get('c.tif').isDir,"
            "};",
        )

        self.assertTrue(rendered["rootIsDir"])
        self.assertTrue(rendered["aIsDir"])
        self.assertEqual(rendered["aChildren"], ["b", "d.tif"])
        self.assertTrue(rendered["nestedIsDir"])
        self.assertFalse(rendered["leafIsDir"])


class StatusBarRenderingTests(WebRenderingTestCase):
    def test_status_bar_reports_workspace_and_token_totals(self):
        rendered = self.run_module(
            module_url("components", "status-bar.js"),
            "const { state } = await import(" + json.dumps(module_url("store.js")) + ");"
            "state.active_workspace = 'demo';"
            "state.cells = [{ id: '1', usage: {"
            "  input_tokens: 2000, output_tokens: 500, requests: 3,"
            "  cache_read_tokens: 1000, cost: 0.01 } }];"
            "const bar = mod.renderStatusBar();"
            "return { id: bar.attributes.id, text: bar.textContent };",
        )

        self.assertEqual(rendered["id"], "status-bar")
        self.assertIn("demo", rendered["text"])
        self.assertIn("↑ 2.0k", rendered["text"])
        self.assertIn("↓ 500", rendered["text"])
        self.assertIn("3 req", rendered["text"])
        self.assertIn("$0.01", rendered["text"])
        self.assertIn("50% cached", rendered["text"])

    def test_status_bar_without_usage_shows_no_totals(self):
        rendered = self.run_module(
            module_url("components", "status-bar.js"),
            "const { state } = await import(" + json.dumps(module_url("store.js")) + ");"
            "state.active_workspace = null;"
            "state.cells = [];"
            "const bar = mod.renderStatusBar();"
            "return { ws: bar.querySelector('.status-ws').textContent,"
            "  stats: bar.querySelectorAll('.status-stats').length,"
            "  toggles: bar.querySelectorAll('.follow-toggle').length };",
        )

        self.assertEqual(rendered["ws"], "No workspace")
        self.assertEqual(rendered["stats"], 0)
        self.assertEqual(rendered["toggles"], 1)

    def test_refresh_status_bar_replaces_only_the_status_bar_node(self):
        rendered = self.run_module(
            module_url("components", "status-bar.js"),
            "const { state } = await import(" + json.dumps(module_url("store.js")) + ");"
            "state.active_workspace = 'demo';"
            "state.cells = [];"
            "const shell = document.createElement('div');"
            "const bar = mod.renderStatusBar();"
            "shell.append(bar);"
            "document.body.append(shell);"
            "mod.refreshStatusBar();"
            "return {"
            "  replaced: document.getElementById('status-bar') !== bar,"
            "  shellChildren: shell.children.length,"
            "  bars: document.querySelectorAll('#status-bar').length,"
            "  ws: document.getElementById('status-bar').querySelector('.status-ws').textContent,"
            "  toggles: document.getElementById('status-bar').querySelectorAll('.follow-toggle').length,"
            "};",
        )

        self.assertTrue(rendered["replaced"])
        self.assertEqual(rendered["shellChildren"], 1)
        self.assertEqual(rendered["bars"], 1)
        self.assertEqual(rendered["ws"], "demo")
        self.assertEqual(rendered["toggles"], 1)


class DataTabRenderingTests(WebRenderingTestCase):
    def test_without_a_workspace_the_tab_shows_the_empty_hint(self):
        rendered = self.run_module(
            module_url("pages", "data.js"),
            "const { state } = await import(" + json.dumps(module_url("store.js")) + ");"
            "state.active_workspace = null;"
            "const tab = mod.renderDataTab();"
            "return { hint: tab.querySelector('.empty-hint').textContent,"
            "  trees: tab.querySelectorAll('.file-tree').length };",
        )

        self.assertIn("No workspace open", rendered["hint"])
        self.assertEqual(rendered["trees"], 0)

    def test_workspace_tab_offers_imports_and_renders_the_file_tree(self):
        rendered = self.run_module(
            module_url("pages", "data.js"),
            "const { state } = await import(" + json.dumps(module_url("store.js")) + ");"
            "state.active_workspace = 'demo';"
            "state.files = ['zones/a.tif', 'roads.geojson'];"
            "const tab = mod.renderDataTab();"
            "const inputs = tab.querySelectorAll('input');"
            "return {"
            "  heading: tab.querySelector('.data-heading-label').textContent,"
            "  refresh: Boolean(tab.querySelector('.data-refresh')),"
            "  hints: tab.querySelectorAll('.empty-hint').length,"
            "  multiple: inputs.filter((i) => i.attributes.multiple === 'multiple').length,"
            "  folder: inputs.filter((i) => 'webkitdirectory' in i.attributes).length,"
            "  urlInput: tab.querySelector('input[type=\"url\"]').attributes.placeholder,"
            "  buttons: [...tab.querySelectorAll('.import-row button')]"
            "    .map((button) => button.textContent),"
            "  treeNames: [...tab.querySelectorAll('.tree-name')].map((s) => s.textContent),"
            "};",
        )

        self.assertEqual(rendered["heading"], "Workspace files")
        self.assertTrue(rendered["refresh"])
        self.assertEqual(rendered["hints"], 0)
        self.assertEqual(rendered["multiple"], 1)
        self.assertEqual(rendered["folder"], 1)
        self.assertEqual(rendered["urlInput"], "https://example.com/file.tif")
        self.assertIn("Import files…", rendered["buttons"])
        self.assertIn("Import folder…", rendered["buttons"])
        self.assertIn("Download", rendered["buttons"])
        self.assertIn("zones", rendered["treeNames"])
        self.assertIn("roads.geojson", rendered["treeNames"])

    def test_import_url_without_a_url_warns_instead_of_requesting(self):
        rendered = self.run_module(
            module_url("pages", "data.js"),
            "const { state } = await import(" + json.dumps(module_url("store.js")) + ");"
            "state.active_workspace = 'demo';"
            "state.files = [];"
            "const tab = mod.renderDataTab();"
            "const rows = tab.querySelectorAll('.import-row');"
            "const download = rows[rows.length - 1].querySelector('button');"
            "download.click();"
            "return { toast: (document.getElementById('toast') || { textContent: '' }).textContent };",
        )

        self.assertEqual(rendered["toast"], "Enter a URL first")


if __name__ == "__main__":
    unittest.main()
