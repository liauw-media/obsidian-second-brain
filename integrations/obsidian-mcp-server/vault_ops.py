"""Vault operations for the Obsidian Second Brain MCP server.

Pure stdlib, no MCP dependency, so the logic is unit-testable on its own. The
MCP wiring in `server.py` is a thin layer over these functions.

Every write follows the AI-first rule (references/ai-first-rules.md): frontmatter
with type/date/tags/ai-first, a `## For future Claude` preamble, and a
`source: mcp` marker so notes added through the connector are distinguishable.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple

_VAULT_ENV = "OBSIDIAN_VAULT_PATH"

# Notes added via the connector land here, separate from hand-authored notes.
_NOTES_DIR = "Inbox"

# Never scanned during search (config, vcs, immutable sources, exports). `.claude`
# is a vault-local agent config dir (CLAUDE.md, commands, settings) - its markdown
# is not vault content and would inflate every result (see issue #80).
_SKIP_DIRS = {".obsidian", ".git", ".trash", ".claude", "_export", "templates"}

# Operational logs and immutable raw sources are rarely the *answer* to a query:
# they are long and term-dense, so without a penalty they dominate term-frequency
# ranking and bury short canonical notes (measured: 0% recall@10 before this - see
# scripts/eval/retrieval_eval.py). De-weight them so they stay findable but cannot
# outrank a real wiki note on equal terms.
# Common function words carry no retrieval signal but recur thousands of times in
# long notes, so without filtering they let any long note outscore the right one on
# "the/what/status" alone (measured: a query like "what is the status of X" returned
# 10 meeting notes, none the target). Drop them from query terms before scoring.
_STOPWORDS = frozenset(
    "the a an and or but of to for in on at by with from as is are was were be been "
    "being do does did doing have has had this that these those it its their there here "
    "what when where who whom which why how whose will would can could should may might "
    "i you he she we they me him her us them my your his our about into over under than "
    "then so if not no yes all any some more most other into out up down off again".split()
)

_SEARCH_DEWEIGHT_PREFIXES = ("raw/",)
_SEARCH_DEWEIGHT_FILES = {"log.md"}
# Tunable so retrieval changes can be A/B-measured (set to 1.0 to disable the penalty).
_SEARCH_DEWEIGHT_FACTOR = float(os.environ.get("OBSIDIAN_SEARCH_DEWEIGHT", "0.15"))

# BM25-style sublinear-TF + length normalization. Env-toggle for A/B (0 = old raw counts).
_SEARCH_LENGTH_NORM = os.environ.get("OBSIDIAN_SEARCH_LENGTHNORM", "1") != "0"

# Optional semantic (meaning-based) layer. Activates ONLY when a local embedding
# index exists at the vault root AND a local Ollama model is reachable - so it is
# opt-in by setup (build the index with scripts/eval/semantic_search.py --build).
# When present, query results are the Reciprocal-Rank-Fusion of lexical + semantic
# (measured best all-rounder). Any failure (no index, Ollama down, bad response)
# silently falls back to pure lexical, so search never breaks or hangs.
_SEMANTIC_INDEX_FILE = ".obsidian-semantic-index.json"
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
_EMBED_MODEL = os.environ.get("OBSIDIAN_EMBED_MODEL", "mxbai-embed-large")
# Backend mirrors scripts/eval/semantic_search.py: "ollama" (default, local) or
# "openai" (any OpenAI-compatible /v1/embeddings - other local runtimes or a cloud API).
_EMBED_BACKEND = os.environ.get("OBSIDIAN_EMBED_BACKEND", "ollama").lower()
_EMBED_URL = os.environ.get("OBSIDIAN_EMBED_URL", _OLLAMA_URL).rstrip("/")
_EMBED_KEY = os.environ.get("OBSIDIAN_EMBED_KEY", "")
_SEMANTIC_ENABLED = os.environ.get("OBSIDIAN_SEARCH_SEMANTIC", "1") != "0"
_RRF_K = 60
_FUSE_DEPTH = 25  # how many from each ranking feed the fusion

# Bounds keep search fast and reads safe.
_MAX_FILES_SCANNED = 2000
_MAX_FILE_BYTES = 200_000
_SNIPPET_CHARS = 320
_READ_CAP = 20_000


def resolve_vault() -> Path:
    """Return the configured vault dir, or raise with a clear message."""
    raw = os.environ.get(_VAULT_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{_VAULT_ENV} is not set")
    vault = Path(raw).expanduser().resolve()
    if not vault.is_dir():
        raise RuntimeError(f"vault path does not exist: {vault}")
    return vault


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed_query(text: str) -> Optional[List[float]]:
    """One fast embedding call for the query via the configured backend. Short
    timeout, no retries - search must stay snappy; on any failure the caller falls
    back to lexical. Supports Ollama (default) and OpenAI-compatible endpoints."""
    if _EMBED_BACKEND == "openai":
        headers = {"Content-Type": "application/json"}
        if _EMBED_KEY:
            headers["Authorization"] = f"Bearer {_EMBED_KEY}"
        body = json.dumps({"model": _EMBED_MODEL, "input": text[:1200]}).encode()
        url = f"{_EMBED_URL}/v1/embeddings"
    else:
        body = json.dumps({"model": _EMBED_MODEL, "prompt": text[:1200], "keep_alive": "15m"}).encode()
        url = f"{_EMBED_URL}/api/embeddings"
    req = urllib.request.Request(url, data=body, headers=headers if _EMBED_BACKEND == "openai"
                                else {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    if data.get("embedding"):
        return data["embedding"]
    items = data.get("data")
    if items and isinstance(items, list) and items[0].get("embedding"):
        return items[0]["embedding"]
    return None


def _passage(vault: Path, rel: str, off: int) -> str:
    """Snippet taken from the chunk that MATCHED, not from the file head.

    `off` is a CHARACTER offset - the index builder slices the decoded string, so
    the file must be read as text and sliced by characters. Byte-seeking would
    mis-align on any multi-byte character (this vault is full of German umlauts
    and box-drawing). Deliberately does NOT use _read_safe(): its _MAX_FILE_BYTES
    cap of 200 KB is smaller than the largest chunk-level notes (up to 338 KB), so
    it would silently return nothing for exactly the documents this exists for.
    """
    try:
        t = (vault / rel).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return " ".join(t[off:off + _SNIPPET_CHARS].split())


def _semantic_fuse(
    query: str, lexical: List[Dict[str, Any]], vault: Path, limit: int
) -> Optional[List[Dict[str, Any]]]:
    """Fuse lexical results with local semantic ranking via RRF. Returns None (so the
    caller uses pure lexical) whenever semantic is unavailable or anything fails."""
    if not _SEMANTIC_ENABLED:
        return None
    index_path = vault / _SEMANTIC_INDEX_FILE
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        notes = index.get("notes") or {}
        if not notes:
            return None
        qvec = _embed_query(query)
        if not qvec:
            return None
        # CHUNK-LEVEL RETRIEVAL (2026-07-30). Long notes carry an extra vector per
        # chunk under n["chunks"]; score them by their BEST-matching passage rather
        # than the mean-pooled document centroid. Reading .get("chunks") means an
        # index WITHOUT chunks behaves exactly as before - the format is additive.
        def _best(n):
            """(score, offset) - offset is None when the note vector won."""
            best, off = _cosine(qvec, n.get("vec") or []), None
            for c in (n.get("chunks") or ()):
                sc = _cosine(qvec, c.get("vec") or [])
                if sc > best:
                    best, off = sc, c.get("off")
            return best, off
        _scored = []
        for rel, n in notes.items():
            if not n.get("vec"):
                continue
            sc, off = _best(n)
            _scored.append({"path": rel, "title": n.get("title", rel), "score": sc, "off": off})
        sem = sorted(_scored, key=lambda r: r["score"], reverse=True)[:_FUSE_DEPTH]
        lex_rank = {r["path"]: i for i, r in enumerate(lexical[:_FUSE_DEPTH])}
        sem_rank = {r["path"]: i for i, r in enumerate(sem)}
        snippet = {r["path"]: r.get("snippet", "") for r in lexical}
        title = {r["path"]: r["title"] for r in lexical}
        for r in sem:
            title.setdefault(r["path"], r["title"])
        match_off = {r["path"]: r["off"] for r in sem if r.get("off") is not None}
        fused = []
        for p in set(lex_rank) | set(sem_rank):
            s = (1.0 / (_RRF_K + lex_rank[p]) if p in lex_rank else 0.0) \
                + (1.0 / (_RRF_K + sem_rank[p]) if p in sem_rank else 0.0)
            snip = snippet.get(p, "")
            mo = match_off.get(p)
            if mo is not None:
                # Prefer the passage that actually matched. Without this a hit deep
                # inside a 338 KB standard is presented with the document's opening
                # lines, which say nothing about why it ranked.
                ps = _passage(vault, p, mo)
                if ps:
                    snip = ps
            rec = {"path": p, "title": title.get(p, p), "snippet": snip, "score": s}
            if mo is not None:
                rec["match_off"] = mo
            fused.append(rec)
        fused.sort(key=lambda r: r["score"], reverse=True)
        out = fused[:limit]
        for r in out:
            r.pop("score", None)
        return out
    except Exception:
        return None  # any failure -> pure lexical, never break search


def search(query: str, *, limit: int = 6) -> List[Dict[str, Any]]:
    """Bounded keyword search over vault markdown, fused with local semantic search
    when an embedding index + Ollama are available (else pure lexical)."""
    vault = resolve_vault()
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2 and t not in _STOPWORDS]
    if not terms:
        # Query was all stopwords/short tokens - fall back to the raw terms so a
        # search like "the who" still returns something rather than nothing.
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    if not terms:
        return []
    limit = max(1, min(int(limit), 20))
    scored: List[Dict[str, Any]] = []
    for i, md in enumerate(_iter_notes(vault)):
        if i >= _MAX_FILES_SCANNED:
            break
        text = _read_safe(md, limit=_MAX_FILE_BYTES)
        if not text:
            continue
        low = text.lower()
        title_low = md.stem.lower()
        # Sublinear term frequency + length normalization (BM25-style): a note that
        # repeats a term 50 times in passing should not outrank a short note that has
        # the term in its title. log1p saturates repeated mentions; dividing the body
        # contribution by a length factor stops long notes winning on sheer volume.
        # Title matches stay a strong, length-independent signal.
        length_norm = 1.0 + math.log1p(len(low) / 1000.0)
        title_score = 0.0
        body_score = 0.0
        for t in terms:
            tc = title_low.count(t)
            if tc:
                title_score += 5.0 * (1.0 + math.log1p(tc))
            bc = low.count(t)
            if bc:
                body_score += 1.0 + math.log1p(bc)
        score = title_score + (body_score / length_norm) if _SEARCH_LENGTH_NORM else float(
            sum(low.count(t) + 5 * title_low.count(t) for t in terms)
        )
        if score:
            rel = md.relative_to(vault).as_posix()
            if rel in _SEARCH_DEWEIGHT_FILES or rel.startswith(_SEARCH_DEWEIGHT_PREFIXES):
                score *= _SEARCH_DEWEIGHT_FACTOR
            scored.append(
                {
                    "path": rel,
                    "title": md.stem,
                    "score": score,
                    "snippet": _snippet(text, terms),
                }
            )
    scored.sort(key=lambda r: r["score"], reverse=True)
    fused = _semantic_fuse(query, scored, vault, limit)
    if fused is not None:
        return fused
    for r in scored:
        r.pop("score", None)
    return scored[:limit]


def read_note(rel: str) -> Dict[str, Any]:
    """Read a note by vault-relative path. Guards against escaping the vault."""
    vault = resolve_vault()
    rel = (rel or "").strip()
    if not rel:
        return {"error": "path is required"}
    target = (vault / rel).resolve()
    if vault != target and vault not in target.parents:
        return {"error": "path is outside the vault"}
    text = _read_safe(target)
    if text is None:
        return {"error": f"not found: {rel}"}
    return {"path": rel, "content": text[:_READ_CAP]}


def save_note(
    title: str,
    content: str,
    *,
    note_type: str = "note",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write an AI-first note to the vault's Inbox folder."""
    vault = resolve_vault()
    title = (title or "").strip()
    content = (content or "").strip()
    if not title or not content:
        return {"error": "title and content are required"}
    note_type = (note_type or "note").strip() or "note"
    tags = [str(t) for t in (tags or [note_type])]

    inbox = vault / _NOTES_DIR
    inbox.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    path = inbox / f"{date} - {_slug(title)}.md"
    tag_block = "\n".join(f"  - {t}" for t in tags)
    preamble = content.split("\n", 1)[0][:280]
    body = (
        f"---\n"
        f"type: {note_type}\n"
        f"date: {date}\n"
        f"tags:\n{tag_block}\n"
        f"ai-first: true\n"
        f"source: mcp\n"
        f"---\n\n"
        f"## For future Claude\n"
        f"{preamble}\n\n"
        f"{content}\n"
    )
    path.write_text(body, encoding="utf-8")
    return {"saved": str(path.relative_to(vault))}


def capture_idea(text: str, *, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Quick idea capture: a lightweight idea note (type: idea) to the Inbox."""
    text = (text or "").strip()
    if not text:
        return {"error": "text is required"}
    title = text.split("\n", 1)[0][:60]
    return save_note(title, text, note_type="idea", tags=tags or ["idea", "capture"])


def update_note(
    rel: str,
    *,
    append: Optional[str] = None,
    heading: Optional[str] = None,
    set_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Guarded edit of an EXISTING vault note.

    Deliberately conservative: it appends a section and/or merges scalar
    frontmatter fields, preserving the rest of the note verbatim. It never
    creates a note (use save_note), never rewrites the body, never touches list
    frontmatter (e.g. `tags:` blocks), and refuses paths outside the vault or in
    protected dirs. Every update stamps `updated: <today>` for provenance.
    """
    vault = resolve_vault()
    rel = (rel or "").strip()
    if not rel:
        return {"error": "path is required"}
    target = (vault / rel).resolve()
    if vault != target and vault not in target.parents:
        return {"error": "path is outside the vault"}
    if set(target.relative_to(vault).parts) & _SKIP_DIRS:
        return {"error": "path is in a protected directory"}
    text = _read_safe(target)
    if text is None:
        return {"error": f"not found: {rel} (update_note only edits existing notes)"}
    if not append and not set_fields:
        return {"error": "nothing to update: provide append and/or set_fields"}

    fm_lines, body, _ = _split_frontmatter(text)
    fields = {str(k): str(v) for k, v in (set_fields or {}).items()}
    fields.setdefault("updated", datetime.now().strftime("%Y-%m-%d"))
    fm_lines = _apply_fields(fm_lines, fields)

    new_body = body
    if append:
        section = append.strip()
        if heading:
            new_body = new_body.rstrip() + f"\n\n## {heading.strip()}\n\n{section}\n"
        else:
            new_body = new_body.rstrip() + f"\n\n{section}\n"

    out = "---\n" + "\n".join(fm_lines).strip("\n") + "\n---\n\n" + new_body.lstrip("\n")
    target.write_text(out, encoding="utf-8")
    return {"updated": rel, "set": sorted(fields.keys()), "appended": bool(append)}


def validate_note(rel: str) -> Dict[str, Any]:
    """Check a note against the AI-first rule and for unresolved wikilinks.

    Returns {path, ok, issues}. Issues cover missing frontmatter, missing
    required keys (type/date/tags/ai-first), a missing `## For future Claude`
    preamble, and `[[wikilinks]]` whose target note does not exist in the vault.
    """
    vault = resolve_vault()
    rel = (rel or "").strip()
    if not rel:
        return {"error": "path is required"}
    target = (vault / rel).resolve()
    if vault != target and vault not in target.parents:
        return {"error": "path is outside the vault"}
    text = _read_safe(target)
    if text is None:
        return {"error": f"not found: {rel}"}

    issues: List[str] = []
    fm_lines, _, had_fm = _split_frontmatter(text)
    fmtext = "\n".join(fm_lines)
    if not had_fm:
        issues.append("missing frontmatter block")
    for key in ("type", "date", "tags", "ai-first"):
        if not re.search(rf"(?mi)^{key}:", fmtext):
            issues.append(f"missing frontmatter key: {key}")
    if "## For future Claude" not in text:
        issues.append("missing '## For future Claude' preamble")
    index = _stem_index(vault)
    seen = set()
    for link in _wikilinks(text):
        norm = _norm_link(link)
        if norm and norm not in index and norm not in seen:
            seen.add(norm)
            issues.append(f"unresolved wikilink: [[{link}]]")
    return {"path": rel, "ok": not issues, "issues": issues}


def backlinks(target: str) -> Dict[str, Any]:
    """Find every note that links to `target` via [[wikilink]].

    `target` may be a note title/stem or a vault-relative path; both resolve to
    the note's stem for matching (aliases `[[Note|alias]]` and folder-qualified
    links `[[folder/Note]]` are handled).
    """
    vault = resolve_vault()
    key = (target or "").strip()
    if not key:
        return {"error": "target is required"}
    stem = _norm_link(Path(key).name if "/" in key or key.endswith(".md") else key)
    refs: List[str] = []
    for i, md in enumerate(_iter_notes(vault)):
        if i >= _MAX_FILES_SCANNED:
            break
        text = _read_safe(md, limit=_MAX_FILE_BYTES) or ""
        if any(_norm_link(link) == stem for link in _wikilinks(text)):
            rel = str(md.relative_to(vault))
            if md.stem.lower() != stem:  # don't list the note itself
                refs.append(rel)
    return {"target": stem, "count": len(refs), "backlinks": sorted(refs)}


def vault_health() -> Dict[str, Any]:
    """Bounded structural health summary of the vault.

    Reports counts plus capped samples of orphan notes (no inbound or outbound
    links), wanted notes (a link exists but the target note does not - a
    wishlist, not an error), and notes with no frontmatter. Bounded by the same
    file cap as search so it stays fast.
    """
    vault = resolve_vault()
    index = _stem_index(vault)
    note_paths: Dict[str, str] = {}
    missing_fm: List[str] = []
    wanted: List[Dict[str, str]] = []
    linked_to: set = set()
    has_outbound: set = set()
    count = 0
    for i, md in enumerate(_iter_notes(vault)):
        if i >= _MAX_FILES_SCANNED:
            break
        count += 1
        rel = str(md.relative_to(vault))
        note_paths[md.stem.lower()] = rel
        text = _read_safe(md, limit=_MAX_FILE_BYTES) or ""
        if not text.lstrip().startswith("---"):
            if len(missing_fm) < 10:
                missing_fm.append(rel)
        links = _wikilinks(text)
        if links:
            has_outbound.add(md.stem.lower())
        for link in links:
            norm = _norm_link(link)
            linked_to.add(norm)
            if norm and norm not in index and len(wanted) < 10:
                wanted.append({"in": rel, "link": link})
    orphans = [p for s, p in note_paths.items() if s not in linked_to and s not in has_outbound]
    return {
        "notes_scanned": count,
        "capped": count >= _MAX_FILES_SCANNED,
        "orphans": {"count": len(orphans), "sample": sorted(orphans)[:10]},
        "wanted_notes": {"count": len(wanted), "sample": wanted},
        "missing_frontmatter": {"count": len(missing_fm), "sample": missing_fm},
    }


# Commands not worth exposing over MCP: meta/setup, Claude-only Google Calendar
# connector commands, and the niche ones flagged on Issue #60 (challenge, health).
_EXCLUDED_SKILLS = {
    "create-command",
    "obsidian-init",
    "obsidian-export",
    "obsidian-visualize",
    "obsidian-challenge",
    "obsidian-health",
    "obsidian-calendar",
    "obsidian-agenda",
    "obsidian-meeting",
    "obsidian-schedule",
}


def list_skills() -> List[Dict[str, Any]]:
    """List the obsidian-second-brain commands exposable as skills (name + description)."""
    cmds = _commands_dir()
    if cmds is None or not cmds.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for md in sorted(cmds.glob("*.md")):
        name = md.stem
        if name in _EXCLUDED_SKILLS:
            continue
        meta, _ = _parse_command(md)
        out.append(
            {
                "name": name,
                "description": meta.get("description", ""),
                "category": meta.get("category", ""),
            }
        )
    return out


def get_skill(name: str) -> Dict[str, Any]:
    """Return a command's playbook (instructions) so the agent can run the skill."""
    name = (name or "").strip().lstrip("/")
    if not name:
        return {"error": "name is required"}
    # Path-traversal guard: skill names are flat slugs (alphanumerics, '-' and '_').
    # Rejecting separators and dots stops a crafted name like "../../etc/passwd" from
    # escaping the commands/ dir, since lstrip("/") alone does not remove ".." segments.
    if not all(c.isalnum() or c in "-_" for c in name):
        return {"error": f"unknown skill: {name}"}
    if name in _EXCLUDED_SKILLS:
        return {"error": f"skill '{name}' is not exposed over MCP"}
    cmds = _commands_dir()
    md = (cmds / f"{name}.md") if cmds else None
    if md is None or not md.is_file():
        return {"error": f"unknown skill: {name}"}
    meta, body = _parse_command(md)
    note = (
        "Run this skill using the MCP tools on this server for vault I/O: "
        "obsidian_search (find/recall), obsidian_read_note (read), "
        "obsidian_save_note / obsidian_capture (write). Follow the steps below."
    )
    return {
        "name": name,
        "description": meta.get("description", ""),
        "instructions": f"{note}\n\n{body.strip()}",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_notes(vault: Path):
    for md in vault.rglob("*.md"):
        if set(md.relative_to(vault).parts) & _SKIP_DIRS:
            continue
        yield md


def _commands_dir() -> Optional[Path]:
    """Locate the skill's commands/ dir: env override, else repo root relative to this file."""
    env = os.environ.get("OBSIDIAN_COMMANDS_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    # this file: <repo>/integrations/obsidian-mcp-server/vault_ops.py
    candidate = Path(__file__).resolve().parents[2] / "commands"
    return candidate if candidate.is_dir() else None


def _parse_command(md: Path):
    """Split a command file into (frontmatter dict, body). Minimal YAML, no deps."""
    text = _read_safe(md) or ""
    meta: Dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end]
            body = text[end + 4 :]
            for line in fm.splitlines():
                if ":" in line and not line.lstrip().startswith(("-", "#", "[")):
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, body


_FRONTMATTER_RE = re.compile(r"\A\ufeff?\s*---\r?\n.*?\r?\n---[ \t]*\r?\n?", re.S)


def _strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block, if present.

    Why: _snippet() centres on the first query-term hit, and in this vault the
    first hit is very often inside the frontmatter's `tags:`/`aliases:`/`title:`.
    A search for "features" therefore returned
    `--- type: reference date: 2026-07-07 tags: - jointhesip - features …`
    which tells a reader nothing about the note. Frontmatter is metadata, not
    prose, so it is excluded from the snippet window entirely.

    Only a block at the very start counts (a `---` horizontal rule later in the
    body must not be mistaken for a frontmatter terminator), hence the start-of-
    string anchor and the non-greedy body. A stray BOM is tolerated.
    """
    return _FRONTMATTER_RE.sub("", text, count=1)


def _snippet(text: str, terms: List[str]) -> str:
    body = _strip_frontmatter(text)
    if not body.strip():          # a note that is nothing but frontmatter
        body = text
    low = body.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=-1)
    if pos < 0:
        # No term in the prose (it matched only in metadata, or only in the
        # title). Opening prose is still far more informative than YAML.
        return " ".join(body.strip()[:_SNIPPET_CHARS].split())
    start = max(0, pos - _SNIPPET_CHARS // 2)
    # Snap the window start to a word boundary so the snippet does not open
    # mid-word ("ck ([[backup-remediation..." instead of "...re-check)").
    # Bounded lookahead: never skip past the matched term itself.
    if start > 0:
        nxt = body.find(" ", start)
        if 0 <= nxt < pos:
            start = nxt + 1
    out = " ".join(body[start : start + _SNIPPET_CHARS].split())
    return ("… " + out) if start > 0 else out


def _read_safe(path: Path, *, limit: int = 4_000_000) -> Optional[str]:
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return None


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:80] or "untitled"


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")


def _wikilinks(text: str) -> List[str]:
    """Return the raw target of each [[wikilink]] (before any | alias or # anchor)."""
    return [m.group(1).strip() for m in _WIKILINK_RE.finditer(text)]


def _norm_link(link: str) -> str:
    """Normalize a wikilink target to a comparable note stem (basename, lowercased)."""
    return link.split("/")[-1].strip().lower()


def _stem_index(vault: Path) -> Dict[str, str]:
    """Map every note's lowercased stem to its vault-relative path (bounded)."""
    idx: Dict[str, str] = {}
    for i, md in enumerate(_iter_notes(vault)):
        if i >= _MAX_FILES_SCANNED:
            break
        idx[md.stem.lower()] = str(md.relative_to(vault))
    return idx


def _split_frontmatter(text: str):
    """Split into (frontmatter_lines, body, had_frontmatter).

    frontmatter_lines excludes the --- fences; body is everything after them.
    Preserves raw lines so unknown/list keys survive a round-trip untouched.
    """
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end].strip("\n")
            body = text[end + 4 :]
            return fm.splitlines(), body, True
    return [], text, False


def _apply_fields(fm_lines: List[str], fields: Dict[str, str]) -> List[str]:
    """Set/replace scalar frontmatter keys, preserving every other line as-is."""
    lines = list(fm_lines)
    remaining = dict(fields)
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Za-z0-9_-]+):", line)
        if m and m.group(1) in remaining:
            key = m.group(1)
            lines[i] = f"{key}: {remaining.pop(key)}"
    for k, v in remaining.items():
        lines.append(f"{k}: {v}")
    return lines


# ---------------------------------------------------------------------------
# heal_links - repoint unresolved [[wikilinks]] at the notes they meant
# ---------------------------------------------------------------------------
#
# A vault links notes by human Title ([[Host iptables rules]]) while the file on
# disk is kebab-cased (host-iptables-rules.md). Those links do not resolve, so
# vault_health counts them as "wanted notes". This repoints each one at the real
# note when exactly one is an unambiguous match, preserving the readable text as
# a display alias: [[host-iptables-rules|Host iptables rules]].
#
# Ported from scripts/heal_links.py, which healed 1715 links across 155 notes on
# 2026-07-08. Two guards are NEW here, because both produce false positives that
# the standalone script did not have to care about:
#   * documentation files talk ABOUT wikilinks - vault_health currently reports
#     {"in": "README.md", "link": "wikilinks"} - and "fixing" those corrupts the
#     vault's own docs;
#   * links inside ``` fences are code samples, not references.

# Vault-ROOT docs only. These talk ABOUT wikilinks, so their [[links]] are
# examples - vault_health really does report {"in": "README.md", "link":
# "wikilinks"}. Matched on the RELATIVE path, never the bare filename: a
# filename match also skipped notes/knowledge-web/*/index.md, which are MOCs
# full of genuine links.
_HEAL_SKIP_RELS = {"readme.md", "agents.md", "_claude.md", "index.md"}
_HEAL_PLACEHOLDER = set("*{}<>")
_HEAL_FENCE_RE = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.S)
_HEAL_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _heal_slug(text: str) -> str:
    """Collapse every non-alphanumeric run to one space, lowercased.

    Makes "Title Case", "kebab-case" and "Flat /24 LAN" compare equal - which is
    exactly what resolves a human-titled wikilink to its kebab-cased file.
    """
    return _HEAL_NON_ALNUM.sub(" ", text.lower()).strip()


def _heal_index(vault: Path):
    """Return (exact_stem -> rel, [stems], slug -> {rels}).

    slug maps to a SET so a collision reads as ambiguous rather than silently
    picking one. Aliases are not indexed (stem-only), so a note reachable only by
    a frontmatter alias reports as ambiguous/no_target rather than being guessed.
    """
    name_to_rel: Dict[str, str] = {}
    slug_to_rels: Dict[str, set] = {}
    for i, md in enumerate(_iter_notes(vault)):
        if i >= _MAX_FILES_SCANNED:
            break
        rel = str(md.relative_to(vault))
        name_to_rel.setdefault(md.stem.lower(), rel)
        slug_to_rels.setdefault(_heal_slug(md.stem), set()).add(rel)
    return name_to_rel, sorted(name_to_rel), slug_to_rels


def _heal_classify(link: str, name_to_rel, stems, slug_to_rels):
    """-> (kind, target) where kind is resolved|fix|ambiguous|no_target|skip."""
    base = link.split("#")[0].split("|")[0].strip()
    if "/" in base:
        base = base.split("/")[-1]
    if not base:
        return "skip", None
    if any(c in link for c in _HEAL_PLACEHOLDER):
        return "skip", None
    low = base.lower()
    if low in name_to_rel:
        return "resolved", name_to_rel[low]
    rels = slug_to_rels.get(_heal_slug(link))
    if rels:
        return ("fix", next(iter(rels))) if len(rels) == 1 else ("ambiguous", sorted(rels))
    near = get_close_matches(low, stems, n=2, cutoff=0.84)
    if len(near) == 1:
        return "fix", name_to_rel[near[0]]
    if len(near) > 1:
        return "ambiguous", [name_to_rel[n] for n in near]
    return "no_target", None


def _heal_rewrite(text: str, link: str, new_stem: str) -> Tuple[str, int]:
    """Repoint [[link]] and [[link|alias]], never touching fenced code.

    [[link]]        -> [[new_stem|link]]   (original text kept as the display)
    [[link|alias]]  -> [[new_stem|alias]]  (author's display text wins)
    """
    # If the link and target differ only by punctuation/case, an alias would just
    # re-display the typo (e.g. a stray trailing backslash from a table cell), so
    # emit the plain link instead.
    plain = _heal_slug(link) == _heal_slug(new_stem)
    bare = f"[[{link}]]"
    aliased = re.compile(r"\[\[" + re.escape(link) + r"\|([^\]]*)\]\]")
    total = 0
    out = []
    for seg in _HEAL_FENCE_RE.split(text):
        if seg.startswith("```") or seg.startswith("~~~") or (
            seg.startswith("`") and seg.endswith("`")
        ):
            out.append(seg)  # code - never rewrite
            continue
        total += seg.count(bare)
        seg = seg.replace(bare, f"[[{new_stem}]]" if plain else f"[[{new_stem}|{link}]]")
        seg, n = aliased.subn(lambda m: f"[[{new_stem}|{m.group(1)}]]", seg)
        total += n
        out.append(seg)
    return "".join(out), total


def heal_links(
    path: Optional[str] = None,
    dry_run: bool = True,
    mapping: Optional[Dict[str, str]] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Repoint unresolved [[wikilinks]] at real notes, alias-preserving.

    DRY RUN BY DEFAULT - returns what it would do and writes nothing. Only an
    explicit dry_run=False writes. Ambiguous links and links with no match are
    reported, never guessed: that judgment belongs to the caller.
    """
    vault = resolve_vault()
    name_to_rel, stems, slug_to_rels = _heal_index(vault)
    overrides = {k.lower(): v for k, v in (mapping or {}).items()}

    if path:
        target = (vault / path).resolve()
        if vault not in target.parents and target != vault:
            raise ValueError("path escapes the vault")
        notes = [target] if target.is_file() else []
    else:
        notes = list(_iter_notes(vault))[:_MAX_FILES_SCANNED]

    fixes: List[Dict[str, str]] = []
    proposals: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    counts = {"resolved": 0, "fix": 0, "ambiguous": 0, "no_target": 0, "skip": 0}
    files_written = 0
    applied = 0

    for md in notes:
        rel = str(md.relative_to(vault))
        if rel.lower() in _HEAL_SKIP_RELS:
            skipped.append({"in": rel, "why": "documentation file - links here are examples"})
            continue
        text = _read_safe(md, limit=_MAX_FILE_BYTES)
        if text is None:
            continue
        scan = _HEAL_FENCE_RE.sub(" ", text)  # ignore links inside code
        per_file: List[Tuple[str, str]] = []
        for link in dict.fromkeys(_wikilinks(scan)):
            if link.lower() in overrides:
                kind, tgt = "fix", overrides[link.lower()]
            else:
                kind, tgt = _heal_classify(link, name_to_rel, stems, slug_to_rels)
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "fix":
                if "templates" in [p.lower() for p in str(tgt).split("/")]:
                    skipped.append({"in": rel, "link": link, "why": "target is a template"})
                    continue
                per_file.append((link, Path(tgt).stem))
                fixes.append({"in": rel, "link": link, "target": Path(tgt).stem})
            elif kind == "ambiguous":
                proposals.append({"in": rel, "link": link, "candidates": tgt})
        if per_file and not dry_run and applied < limit:
            new = text
            changed = 0
            for link, stem in per_file:
                new, n = _heal_rewrite(new, link, stem)
                changed += n
            if changed and new != text:
                md.write_text(new, encoding="utf-8")
                files_written += 1
                applied += changed

    return {
        "dry_run": dry_run,
        "counts": counts,
        "fixes": fixes[:limit],
        "proposals": proposals[:limit],
        "skipped": skipped[:limit],
        "files_written": files_written,
        "links_rewritten": applied,
    }
