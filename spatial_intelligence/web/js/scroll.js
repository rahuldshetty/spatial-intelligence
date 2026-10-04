/* Follow the newest output, on the reader's terms.

The panel that holds the cells is rebuilt on every shell render, so the follow
state cannot live on the element: it lives here, the live container is looked up
by id whenever it is needed, and the scroll listener moves to whichever element
answers to that id now. One button, in the panel, turns following on.

Following is off until the reader asks for it. While on, the panel pins to the
bottom as content arrives; a scroll the app did not make — the reader — turns it
off again.
*/

"use strict";

/* How close to the bottom still counts as "at the bottom". The slack absorbs
   fractional layout and the last line's descender. */
const BOTTOM_SLACK = 48;

/* Where our own last write landed, or null. Scroll events arrive a frame later,
   after the content may have grown again; reading that as the reader scrolling
   away is what used to stop the panel from following a fast stream. */
let landed = null;
/* The element the reader listener is attached to; the shell replaces it. */
let bound = null;
let queued = false;
let enabled = false;
let observer = null;

/** The panel's scroll container, if the shell has one right now. */
function panel() {
  return document.getElementById("tab-content");
}

/** Whether the container is scrolled to (or very near) its bottom. */
export function atBottom(container, slack = BOTTOM_SLACK) {
  if (!container) return true;
  return container.scrollHeight - container.scrollTop - container.clientHeight <= slack;
}

export function isFollowing() {
  return enabled;
}

function setFollowing(next) {
  const value = Boolean(next);
  if (value === enabled) return;
  enabled = value;
  syncToggle();
  if (value) hold();
}

/** Stop following: the reader keeps whatever position they scrolled to. */
export function unfollow() {
  setFollowing(false);
}

/** Follow from now on, and go to the bottom of what is already there. */
export function jump() {
  setFollowing(true);
}

/** Follow the newest content, if the reader asked us to. */
export function hold() {
  const container = panel();
  if (!container) return;
  bindReader(container);
  if (!enabled || queued) return;
  queued = true;
  window.requestAnimationFrame(() => {
    queued = false;
    if (!enabled) return;
    const live = panel();
    if (!live) return;
    live.scrollTop = live.scrollHeight;
    landed = live.scrollTop;  // what the browser actually clamped it to
  });
}

/** Move the panel the way the app does; the reader's own scroll is not this.

When following, a stale position is overruled: the panel belongs at the bottom.
*/
export function scrollTo(container, top) {
  if (!container) return;
  if (enabled) {
    hold();
    return;
  }
  container.scrollTop = top;
  landed = container.scrollTop;
}

/** Watch the shell for growth, so the panel follows without a repaint call.

Streamed steps append and re-text nodes as they arrive; watching the subtree
catches every path, so callers never have to remember to ask.
*/
export function watchOutput(root) {
  if (observer || typeof MutationObserver !== "function") return;
  const app = root || document.getElementById("app") || document.body;
  if (!app) return;
  observer = new MutationObserver(() => hold());
  observer.observe(app, { childList: true, subtree: true, characterData: true });
}

/* The one checkbox on screen. The status bar is rebuilt on every update, so this
   holds the newest one and writes the state into it directly rather than
   subscribing each copy — a subscriber per rebuild would never be released. */
let toggle = null;

function syncToggle() {
  if (!toggle) return;
  toggle.checked = enabled;
  if (toggle.parentElement) toggle.parentElement.classList.toggle("active", enabled);
}

/** The status-bar checkbox that turns following on and off. */
export function followToggle() {
  const box = document.createElement("input");
  box.type = "checkbox";
  box.setAttribute("id", "follow-output");
  const label = document.createElement("label");
  label.className = "follow-toggle";
  label.setAttribute("for", "follow-output");
  label.title = "Scroll to the bottom as new output arrives";
  label.append(box, document.createTextNode("Follow output"));
  box.checked = enabled;
  label.classList.toggle("active", enabled);
  box.addEventListener("change", () => setFollowing(box.checked));
  toggle = box;
  return label;
}

/* The panel keeps its id across rebuilds, so the listener has to move to the
   element that answers to it now. Only a scroll the app did not ask for is the
   reader moving; ours says nothing. */
function bindReader(container) {
  if (container === bound) return;
  bound = container;
  landed = null;
  container.addEventListener(
    "scroll",
    () => {
      if (landed !== null && Math.abs(container.scrollTop - landed) <= 1) {
        landed = null;
        return;
      }
      landed = null;
      if (!atBottom(container)) setFollowing(false);
    },
    { passive: true }
  );
}
