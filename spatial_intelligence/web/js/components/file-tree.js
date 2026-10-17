/* Workspace file tree: nested directory view used by the Data tab. */

import { el } from "../dom.js";
import { icon } from "../icons.js";

const IMAGE_EXT = new Set(["tif", "tiff", "png", "jpg", "jpeg", "webp", "vrt", "jp2"]);
const TABULAR_EXT = new Set(["csv", "parquet", "xlsx", "xls", "json", "geojson", "gpkg"]);
const DOC_EXT = new Set(["md", "txt", "pdf", "ipynb", "py", "html"]);
const VECTOR_EXT = new Set(["shp", "geojson", "gpkg", "kml", "gml", "fgb"]);

export function buildFileTree(files) {
  const root = { name: "", isDir: true, children: new Map() };
  for (const f of files) {
    const parts = String(f).split("/");
    let node = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isDir = i < parts.length - 1;
      let child = node.children.get(part);
      if (!child) {
        child = {
          name: part,
          isDir,
          children: new Map(),
          path: parts.slice(0, i + 1).join("/"),
        };
        node.children.set(part, child);
      }
      node = child;
    }
  }
  return root;
}

function sortedChildren(node) {
  return [...node.children.values()].sort((a, b) => {
    if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

function fileIcon(name) {
  const ext = String(name).split(".").pop().toLowerCase();
  if (VECTOR_EXT.has(ext)) return { name: "layers", className: "kind-vector" };
  if (IMAGE_EXT.has(ext)) return { name: "file-image", className: "kind-raster" };
  if (TABULAR_EXT.has(ext)) return { name: "database", className: "kind-table" };
  if (DOC_EXT.has(ext)) return { name: "file", className: "kind-doc" };
  return { name: "file", className: "" };
}

export function renderFileTree(files, actions = {}) {
  const ul = el("ul", { class: "file-tree" });
  for (const node of sortedChildren(buildFileTree(files))) {
    ul.append(renderTreeNode(node, actions));
  }
  return ul;
}

function treeAction(iconName, title, onclick) {
  const button = el("button", {
    type: "button",
    class: "tree-action",
    title,
    "aria-label": title,
  });
  button.append(icon(iconName, { size: 13 }));
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    onclick();
  });
  return button;
}

export function renderTreeNode(node, actions = {}) {
  if (node.isDir) {
    const li = el("li", { class: "tree-dir" });
    const details = el("details", { open: "" });
    const summary = el("summary", {});
    summary.append(
      icon("chevron-right", { size: 12, class: "tree-chevron" }),
      icon("folder", { size: 15, class: "tree-icon kind-folder" }),
      el("span", { class: "tree-name", text: node.name })
    );
    details.append(summary);
    const childUl = el("ul", { class: "file-tree" });
    for (const child of sortedChildren(node)) {
      childUl.append(renderTreeNode(child, actions));
    }
    details.append(childUl);
    li.append(details);
    return li;
  }
  const li = el("li", { class: "tree-file" });
  const meta = fileIcon(node.name);
  li.append(
    icon(meta.name, { size: 14, class: "tree-icon " + meta.className }),
    el("span", { class: "tree-name", text: node.name })
  );
  if (actions.onRename || actions.onDelete) {
    const tools = el("span", { class: "tree-actions" });
    if (actions.onRename) {
      tools.append(
        treeAction("pencil", "Rename " + node.name, () => actions.onRename(node.path))
      );
    }
    if (actions.onDelete) {
      tools.append(
        treeAction("trash-2", "Delete " + node.name, () => actions.onDelete(node.path))
      );
    }
    li.append(tools);
  }
  return li;
}
