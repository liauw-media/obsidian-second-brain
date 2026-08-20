"""Regressions from the 2026-08-16 hamidasiblog batch (#210, #211, #212).

Three research-toolkit bugs with one shape: a default that silently returns
nothing for a whole class of users. /youtube asked only for English
transcripts (#210), the pinned Gemini model 404s for new API keys (#211),
and vault_scan kept the pre-#159 tokenizer that returns zero notes for CJK
topics - the third stale copy of that tokenizer, surviving three fixes
(#212).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("OBSIDIAN_VAULT_PATH", str(REPO_ROOT / "examples" / "sample-vault"))


# ── #212: vault_scan tokenizes via the search tokenizer ─────────────────────

def test_topic_terms_is_cjk_aware():
    from research.lib.vault_terms import topic_terms

    # 2-char Japanese compound: dropped entirely by the old len(w) > 2 filter.
    assert topic_terms("習慣"), "2-char CJK word must produce terms"
    # Unspaced Japanese phrase: one giant token under the old whitespace split.
    terms = topic_terms("朝ラボの習慣")
    assert "習慣" in terms, terms
    assert all(len(t) <= 2 or not any("぀" <= c <= "鿿" for c in t) for t in terms), (
        "CJK runs must be split into bigrams, not kept whole: %s" % terms)
    # English unchanged: meaningful words survive, stopwords drop.
    en = topic_terms("habit routine")
    assert "habit" in en and "routine" in en


def _ensure_genai_stub():
    """notebooklm.py imports google.genai at module top for the File Search
    flow; vault_scan never touches it. CI installs the base deps only, so
    stub the package there rather than skipping the CJK regression test."""
    try:
        import google.genai  # noqa: F401
        return
    except ImportError:
        pass
    google_pkg = sys.modules.setdefault("google", types.ModuleType("google"))
    genai = types.ModuleType("google.genai")
    genai.types = types.ModuleType("google.genai.types")
    google_pkg.genai = genai
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = genai.types


def test_vault_scan_finds_cjk_topic(tmp_path, monkeypatch):
    _ensure_genai_stub()
    from research import notebooklm, research_deep

    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / "wiki" / "morning.md").write_text(
        "---\ntype: note\n---\n\n朝ラボの習慣を記録する。習慣が大事。\n", encoding="utf-8"
    )
    (vault / "wiki" / "unrelated.md").write_text(
        "---\ntype: note\n---\n\nNothing relevant here.\n", encoding="utf-8"
    )
    for mod in (notebooklm, research_deep):
        monkeypatch.setattr(mod, "VAULT_PATH", vault)
        hits = mod.vault_scan("朝ラボの習慣")
        paths = [h["path"] for h in hits]
        assert any("morning.md" in p for p in paths), (mod.__name__, paths)
        assert not any("unrelated.md" in p for p in paths), (mod.__name__, paths)


def test_no_stale_tokenizer_copies_left_in_command_paths():
    """The private copy survived #159, #188 and #192 because nothing fenced it.
    Command-path scripts must not re-grow a whitespace-split + len filter;
    scripts/eval/ keeps its own token semantics on purpose (gold matching)."""
    import re as _re

    # The code shape, not prose mentions in comments: a split feeding a
    # same-line length filter.
    pattern = _re.compile(r"re\.split\([^)]*\).*len\(\w+\)\s*>\s*\d")
    offenders = []
    for py in (REPO_ROOT / "scripts" / "research").rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            if pattern.search(line):
                offenders.append(f"{py}: {line.strip()}")
    assert not offenders, offenders


# ── #210: /youtube transcript language preference + fallback ────────────────

class _Snip:
    def __init__(self, text):
        self.text = text


class _Fetched:
    def __init__(self, texts):
        self.snippets = [_Snip(t) for t in texts]


def _install_fake_yta(monkeypatch, fetch_impl, list_impl=None):
    fake = types.ModuleType("youtube_transcript_api")

    class FakeApi:
        def fetch(self, video_id, languages=None):
            return fetch_impl(video_id, languages)

        def list(self, video_id):
            if list_impl is None:
                raise AssertionError("list() should not be called")
            return list_impl(video_id)

    fake.YouTubeTranscriptApi = FakeApi
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake)


def test_get_transcript_passes_configured_languages(monkeypatch):
    from research.lib import youtube

    seen = {}

    def fetch(video_id, languages):
        seen["languages"] = languages
        return _Fetched(["こんにちは", "世界"])

    _install_fake_yta(monkeypatch, fetch)
    monkeypatch.setenv("TRANSCRIPT_LANGUAGES", "ja, en")
    assert youtube.get_transcript("vid") == "こんにちは 世界"
    assert seen["languages"] == ["ja", "en"]


def test_get_transcript_falls_back_to_available_language(monkeypatch):
    from research.lib import youtube

    class _Transcript:
        language_code = "ja"

        def fetch(self):
            return _Fetched(["字幕", "テキスト"])

    def fetch(video_id, languages):
        raise RuntimeError("NoTranscriptFound: requested ('en',)")

    _install_fake_yta(monkeypatch, fetch, list_impl=lambda vid: iter([_Transcript()]))
    monkeypatch.delenv("TRANSCRIPT_LANGUAGES", raising=False)
    assert youtube.get_transcript("vid") == "字幕 テキスト"


def test_get_transcript_returns_none_when_nothing_available(monkeypatch):
    from research.lib import youtube

    def fetch(video_id, languages):
        raise RuntimeError("NoTranscriptFound")

    def list_impl(vid):
        raise RuntimeError("TranscriptsDisabled")

    _install_fake_yta(monkeypatch, fetch, list_impl=list_impl)
    assert youtube.get_transcript("vid") is None


# ── #211: Gemini model fallback ladder ───────────────────────────────────────

def _resp(status, text="", payload=None):
    class R:
        status_code = status

        def json(self):
            return payload

    R.text = text
    return R()


def _ok_payload(body_text):
    return {
        "candidates": [{"content": {"parts": [{"text": body_text}]}}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }


@pytest.fixture()
def gemini(monkeypatch):
    from research.lib import gemini as g

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(g.usage, "log_call", lambda *a, **k: None)
    monkeypatch.setattr(g, "_resolved_default", None)
    return g


def test_default_walks_ladder_past_cohort_404(gemini, monkeypatch):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        if gemini.GEMINI_DEFAULT_MODEL in url:
            return _resp(404, "no longer available to new users")
        return _resp(200, payload=_ok_payload("ok"))

    monkeypatch.setattr(gemini.requests, "post", post)
    out = gemini.call("hi", command="test")
    assert out["text"] == "ok"
    assert len(calls) == 2, calls
    assert gemini.MODEL_LADDER[1] in calls[1]
    # The working model is remembered: the next call does not re-probe.
    calls.clear()
    gemini.call("hi again", command="test")
    assert len(calls) == 1 and gemini.MODEL_LADDER[1] in calls[0]


def test_explicit_model_is_never_laddered(gemini, monkeypatch):
    def post(url, json=None, headers=None, timeout=None):
        return _resp(404, "not available")

    monkeypatch.setattr(gemini.requests, "post", post)
    with pytest.raises(RuntimeError) as e:
        gemini.call("hi", command="test", model="gemini-custom")
    msg = str(e.value)
    assert "gemini-custom" in msg
    assert "GEMINI_SUMMARY_MODEL" in msg, "the error must name the fix"


def test_ladder_exhaustion_names_the_env_fix(gemini, monkeypatch):
    def post(url, json=None, headers=None, timeout=None):
        return _resp(404, "not available")

    monkeypatch.setattr(gemini.requests, "post", post)
    with pytest.raises(RuntimeError) as e:
        gemini.call("hi", command="test")
    assert "GEMINI_SUMMARY_MODEL" in str(e.value)
