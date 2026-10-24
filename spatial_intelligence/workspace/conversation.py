"""Per-workspace conversation index: SQLite + FTS5.

The agent's trace files stay the record of a run, but they are append-only JSONL
per cell and not searchable. This module mirrors the parts of a run that answer a
follow-up question -- the prompt, visible assistant notes, each tool call and its
result, and the final answer -- into one SQLite database under
``<workspace>/traces/``. Reasoning text (steps marked ``reasoning``) is
deliberately not indexed, and the streamed answer block is folded into the final
answer so the same text is not stored twice.

Every row records the prompt cell it belongs to (``cell_id``), the run, the
trace position, and the row id of the prompt that introduced it (``prompt_id``),
so a search hit can name both the prompt and the step it came from. Deleting a
prompt cell deletes its rows; re-running one replaces them.

One connection is open per workspace. The session opens it when a workspace
opens and closes it when that workspace is left, so a switch never leaves the
previous workspace's file handle (or WAL) behind.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Maximum characters one indexed unit keeps. A tool result can be a whole
#: file; the index stores enough to find it, not a second copy of it.
MAX_UNIT_CHARS = 120_000

#: Maximum characters of a prompt/answer preview returned with a hit.
PREVIEW_CHARS = 280

#: Maximum number of terms taken from one query.
MAX_QUERY_TERMS = 16

#: Tokens of surrounding text a hit shows.
SNIPPET_TOKENS = 48

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def now_iso() -> str:
    """Return the current UTC time as a second-resolution ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def query_terms(query: str) -> list[str]:
    """Split a natural-language query into unique search terms."""
    seen: set[str] = set()
    terms: list[str] = []
    for word in _WORD.findall(query or ""):
        folded = word.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        terms.append(word)
        if len(terms) >= MAX_QUERY_TERMS:
            break
    return terms


def _clean(value: Any) -> str:
    """Collapse whitespace so a snippet renders on one line."""
    return _WHITESPACE.sub(" ", str(value or "")).strip()


def _clip(text: str) -> str:
    """Bound one unit; the tail is dropped, not summarized."""
    return text[:MAX_UNIT_CHARS]


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _step_text(step: dict) -> tuple[str, str] | None:
    """Return ``(kind, text)`` for an indexable trace step."""
    kind = step.get("type")
    name = step.get("name") or "tool"
    if kind == "tool_call":
        args = step.get("args")
        return "tool_call", _clip(f"{name} called with {_json(args)}")
    if kind == "tool_result":
        content = step.get("content")
        if content is None or content == "":
            return None
        if not isinstance(content, str):
            content = _json(content)
        return "tool_result", _clip(f"{name} result: {content}")
    return None


class ConversationStore:
    """The open workspace's searchable conversation history.

    Safe to call from the run worker and an HTTP request at once: every
    operation takes one re-entrant lock, and the connection is created with
    ``check_same_thread=False``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._path = Path(path) if path is not None else None
        self._conn: sqlite3.Connection | None = None
        # cell_id -> (prompt row id, run id) for the run currently streaming.
        self._active: dict[str, tuple[int | None, str | None]] = {}
        # cell_id -> [first trace index, text chunks] for visible assistant text
        # that has not reached a block boundary yet.
        self._pending_text: dict[str, list] = {}

    # -- lifecycle -------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """The database file, or ``None`` before the first open."""
        return self._path

    @property
    def open(self) -> bool:
        """Whether a connection is currently held."""
        return self._conn is not None

    def open_at(self, path: str | Path) -> None:
        """Open (creating if needed) the database at ``path``.

        Re-opening the file that is already open is a no-op: a workspace can be
        re-opened without dropping the connection and rebuilding the index.
        """
        target = Path(path)
        with self._lock:
            if self._conn is not None and self._path == target:
                return
            self.close()
            target.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(target, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                self._ensure_schema(conn)
            except Exception:
                conn.close()
                raise
            self._conn = conn
            self._path = target
            self._active.clear()

    def close(self) -> None:
        """Close the connection (a later call opens it again)."""
        with self._lock:
            conn = self._conn
            self._conn = None
            self._active.clear()
            self._pending_text.clear()
            if conn is not None:
                conn.close()

    # -- schema ----------------------------------------------------------

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_units (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cell_id TEXT NOT NULL,
                prompt_id INTEGER,
                run_id TEXT,
                kind TEXT NOT NULL,
                step_index INTEGER,
                step_id TEXT,
                tool_name TEXT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversation_units_cell
                ON conversation_units(cell_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(
                text,
                content='conversation_units',
                content_rowid='id',
                tokenize='unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS conversation_units_ai
                AFTER INSERT ON conversation_units BEGIN
                    INSERT INTO conversation_fts(rowid, text)
                    VALUES (new.id, new.text);
                END;
            CREATE TRIGGER IF NOT EXISTS conversation_units_ad
                AFTER DELETE ON conversation_units BEGIN
                    INSERT INTO conversation_fts(conversation_fts, rowid, text)
                    VALUES ('delete', old.id, old.text);
                END;
            """
        )
        conn.commit()

    # -- writes ----------------------------------------------------------

    def begin_run(self, cell_id: str, run_id: str, prompt: str) -> None:
        """Start a run: replace the cell's rows and index its prompt."""
        text = _clip(_clean(prompt))
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            with conn:
                conn.execute("DELETE FROM conversation_units WHERE cell_id=?", (cell_id,))
                self._pending_text.pop(cell_id, None)
                self._active[cell_id] = (None, run_id)
                if not text:
                    return
                cursor = conn.execute(
                    """
                    INSERT INTO conversation_units
                        (cell_id, prompt_id, run_id, kind, step_index, step_id,
                         tool_name, text, created_at)
                    VALUES (?, NULL, ?, 'prompt', NULL, NULL, NULL, ?, ?)
                    """,
                    (cell_id, run_id, text, now_iso()),
                )
                prompt_id = int(cursor.lastrowid)
                conn.execute(
                    "UPDATE conversation_units SET prompt_id=? WHERE id=?",
                    (prompt_id, prompt_id),
                )
                self._active[cell_id] = (prompt_id, run_id)

    def index_resume(self, cell_id: str, run_id: str, answers: dict) -> None:
        """Index the user's answers as the prompt of a resumed run."""
        text = _clip(_clean("User answers: " + _json(answers)))
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            self._flush_text(cell_id, conn)
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO conversation_units
                        (cell_id, prompt_id, run_id, kind, step_index, step_id,
                         tool_name, text, created_at)
                    VALUES (?, NULL, ?, 'prompt', NULL, NULL, NULL, ?, ?)
                    """,
                    (cell_id, run_id, text, now_iso()),
                )
                prompt_id = int(cursor.lastrowid)
                conn.execute(
                    "UPDATE conversation_units SET prompt_id=? WHERE id=?",
                    (prompt_id, prompt_id),
                )
                previous = self._active.get(cell_id)
                self._active[cell_id] = (
                    prompt_id,
                    run_id or (previous or (None, None))[1],
                )

    def index_step(self, cell_id: str, step: dict, step_index: int | None) -> None:
        """Index one trace step: tool calls/results and visible assistant text.

        Assistant text is streamed as many deltas, so it is accumulated and
        stored once per block. Reasoning deltas (``reasoning``) flush any
        visible block but are never stored themselves.
        """
        step = step or {}
        kind = step.get("type")
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            if kind in ("text", "text_delta"):
                self._index_text(cell_id, conn, step, kind, step_index)
                return
            self._flush_text(cell_id, conn)
            parsed = _step_text(step)
            if parsed is None:
                return
            kind, text = parsed
            if not text:
                return
            prompt_id, run_id = self._active.get(cell_id, (None, None))
            with conn:
                conn.execute(
                    """
                    INSERT INTO conversation_units
                        (cell_id, prompt_id, run_id, kind, step_index, step_id,
                         tool_name, text, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cell_id,
                        prompt_id,
                        run_id,
                        kind,
                        step_index,
                        step.get("tool_call_id"),
                        step.get("name"),
                        text,
                        now_iso(),
                    ),
                )

    def _index_text(
        self,
        cell_id: str,
        conn: sqlite3.Connection,
        step: dict,
        kind: str,
        step_index: int | None,
    ) -> None:
        """Accumulate one visible text step; skip model reasoning."""
        if step.get("reasoning"):
            self._flush_text(cell_id, conn)
            return
        content = step.get("content") or ""
        if not content:
            return
        pending = self._pending_text.get(cell_id)
        # A part_start opens a new text part, so anything buffered belongs to
        # the previous one.
        if kind == "text" and pending is not None:
            self._flush_text(cell_id, conn)
            pending = None
        if pending is None:
            self._pending_text[cell_id] = [step_index, [content]]
        else:
            pending[1].append(content)

    def _flush_text(self, cell_id: str, conn: sqlite3.Connection) -> None:
        """Store the accumulated visible text as one ``assistant`` unit."""
        pending = self._pending_text.pop(cell_id, None)
        if pending is None:
            return
        index, chunks = pending
        text = _clip(_clean("".join(chunks)))
        if not text:
            return
        prompt_id, run_id = self._active.get(cell_id, (None, None))
        with conn:
            conn.execute(
                """
                INSERT INTO conversation_units
                    (cell_id, prompt_id, run_id, kind, step_index, step_id,
                     tool_name, text, created_at)
                VALUES (?, ?, ?, 'assistant', ?, NULL, NULL, ?, ?)
                """,
                (cell_id, prompt_id, run_id, index, text, now_iso()),
            )

    def index_result(
        self,
        cell_id: str,
        status: str,
        output: str | None,
        error: str | None,
    ) -> None:
        """Index the final answer (or the failure) of a run."""
        text = output if status == "done" else error or output
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            # The last visible text block is the answer streamed back to the
            # browser; the result below stores the same answer once.
            self._flush_text(cell_id, conn)
            if not text:
                return
            kind = "response" if status == "done" else "error"
            prompt_id, run_id = self._active.get(cell_id, (None, None))
            normalized = _clip(_clean(text))
            with conn:
                conn.execute(
                    """
                    INSERT INTO conversation_units
                        (cell_id, prompt_id, run_id, kind, step_index, step_id,
                         tool_name, text, created_at)
                    VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                    """,
                    (cell_id, prompt_id, run_id, kind, normalized, now_iso()),
                )
            if status == "done":
                self._drop_duplicate_answer(cell_id, conn, normalized)

    @staticmethod
    def _drop_duplicate_answer(
        cell_id: str, conn: sqlite3.Connection, answer: str
    ) -> None:
        """Remove trailing assistant blocks that only repeat the final answer."""
        while True:
            row = conn.execute(
                """
                SELECT id, text FROM conversation_units
                 WHERE cell_id=? AND kind='assistant'
                 ORDER BY id DESC LIMIT 1
                """,
                (cell_id,),
            ).fetchone()
            if row is None or _clean(row["text"]) != answer:
                return
            with conn:
                conn.execute("DELETE FROM conversation_units WHERE id=?", (row["id"],))

    def forget_cell(self, cell_id: str) -> None:
        """Drop every row belonging to one prompt cell."""
        with self._lock:
            self._active.pop(cell_id, None)
            self._pending_text.pop(cell_id, None)
            conn = self._conn
            if conn is None:
                return
            with conn:
                conn.execute("DELETE FROM conversation_units WHERE cell_id=?", (cell_id,))

    # -- reads -----------------------------------------------------------

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Return the best matching units, ranked by BM25 (lower is better)."""
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self._lock:
            conn = self._conn
            if conn is None:
                return []
            rows = conn.execute(
                """
                SELECT u.id, u.cell_id, u.prompt_id, u.run_id, u.kind,
                       u.step_index, u.step_id, u.tool_name,
                       snippet(conversation_fts, 0, '', '', '…', ?) AS hint,
                       bm25(conversation_fts) AS score,
                       (SELECT substr(p.text, 1, ?) FROM conversation_units AS p
                         WHERE p.cell_id = u.cell_id AND p.kind = 'prompt'
                         ORDER BY p.id DESC LIMIT 1) AS prompt,
                       (SELECT substr(a.text, 1, ?) FROM conversation_units AS a
                         WHERE a.cell_id = u.cell_id AND a.kind IN ('response', 'error')
                         ORDER BY a.id DESC LIMIT 1) AS answer
                  FROM conversation_fts
                  JOIN conversation_units AS u ON u.id = conversation_fts.rowid
                 WHERE conversation_fts MATCH ?
                 ORDER BY score
                 LIMIT ?
                """,
                (SNIPPET_TOKENS, PREVIEW_CHARS, PREVIEW_CHARS, match, max(1, int(limit))),
            ).fetchall()
            hits: list[dict[str, Any]] = []
            for row in rows:
                kind = row["kind"]
                tool = row["tool_name"]
                step = row["step_index"]
                reference = f"prompt cell {row['cell_id']} · {kind}"
                if tool:
                    reference = f"{reference} · {tool}"
                if step is not None:
                    reference = f"{reference} · step {step}"
                hits.append(
                    {
                        "reference": reference,
                        "cell_id": row["cell_id"],
                        "prompt_id": row["prompt_id"],
                        "run_id": row["run_id"],
                        "kind": kind,
                        "step_index": step,
                        "step_id": row["step_id"],
                        "tool_name": tool,
                        "hint": _clean(row["hint"]),
                        "prompt": _clean(row["prompt"]),
                        "answer": _clean(row["answer"]),
                        "score": float(row["score"]),
                    }
                )
            return hits

    def unit_count(self, cell_id: str | None = None) -> int:
        """Number of indexed units, for diagnostics and tests."""
        with self._lock:
            conn = self._conn
            if conn is None:
                return 0
            if cell_id is None:
                return int(conn.execute("SELECT count(*) FROM conversation_units").fetchone()[0])
            return int(
                conn.execute(
                    "SELECT count(*) FROM conversation_units WHERE cell_id=?",
                    (cell_id,),
                ).fetchone()[0]
            )


__all__ = [
    "ConversationStore",
    "MAX_UNIT_CHARS",
    "PREVIEW_CHARS",
    "now_iso",
    "query_terms",
]
