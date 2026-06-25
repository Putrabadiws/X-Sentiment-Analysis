"""Tes shim kompatibilitas X: resolver query-ID + patch endpoint + get_indices.

Semua mock jaringan (httpx / fake session) — tak ada panggilan X asli.
"""
import asyncio
import types

import pytest

from src import x_transaction_patch as xtp


def _fake_get(mapping):
    """Pengganti httpx.get: cocokkan substring URL → text. Rekam panggilan."""
    calls = []

    def _get(url, **kwargs):
        calls.append(url)
        for key, text in mapping.items():
            if key in url:
                return types.SimpleNamespace(text=text)
        raise AssertionError(f"URL tak terduga: {url}")

    _get.calls = calls
    return _get


_HOME = (
    'a<script src="https://abs.twimg.com/responsive-web/client-web/vendor.0a0a0a0a.js"></script>'
    '<script src="https://abs.twimg.com/responsive-web/client-web/main.deadbeef.js"></script>'
)
_MAIN_JS = (
    'x={queryId:"BMK",operationName:"BookmarkSearchTimeline"};'  # adversarial: dilewati
    'y={queryId:"ABC123",operationName:"SearchTimeline",operationType:"query"}'
)


@pytest.fixture
def reset_cache():
    xtp._resolved_qid = None
    yield
    xtp._resolved_qid = None


# --- resolver query-ID ----------------------------------------------------------
def test_resolver_extracts_live_qid_and_skips_bookmark(monkeypatch, reset_cache):
    monkeypatch.setattr(xtp.httpx, "get",
                        _fake_get({"x.com/home": _HOME, "main.deadbeef.js": _MAIN_JS}))
    assert xtp.resolve_search_query_id("a", "b") == "ABC123"


def test_resolver_fallback_on_network_error(monkeypatch, reset_cache):
    def boom(*a, **k):
        raise RuntimeError("no net")

    monkeypatch.setattr(xtp.httpx, "get", boom)
    assert xtp.resolve_search_query_id("a", "b") == xtp._FALLBACK_SEARCH_QID


def test_resolver_fallback_when_qid_absent(monkeypatch, reset_cache):
    monkeypatch.setattr(xtp.httpx, "get",
                        _fake_get({"x.com/home": _HOME, "main.deadbeef.js": "kosong"}))
    assert xtp.resolve_search_query_id("a", "b") == xtp._FALLBACK_SEARCH_QID


def test_resolver_caches_result(monkeypatch, reset_cache):
    get = _fake_get({"x.com/home": _HOME, "main.deadbeef.js": _MAIN_JS})
    monkeypatch.setattr(xtp.httpx, "get", get)
    xtp.resolve_search_query_id("a", "b")
    n = len(get.calls)
    xtp.resolve_search_query_id("a", "b")  # dari cache
    assert len(get.calls) == n


def test_resolver_force_bypasses_cache(monkeypatch, reset_cache):
    get = _fake_get({"x.com/home": _HOME, "main.deadbeef.js": _MAIN_JS})
    monkeypatch.setattr(xtp.httpx, "get", get)
    xtp.resolve_search_query_id("a", "b")
    n = len(get.calls)
    xtp.resolve_search_query_id("a", "b", force=True)
    assert len(get.calls) > n


# --- patch endpoint -------------------------------------------------------------
def test_patch_search_endpoint_builds_url():
    xtp.patch_search_endpoint("ZZZ")
    from twikit.client.gql import Endpoint
    assert Endpoint.SEARCH_TIMELINE == "https://x.com/i/api/graphql/ZZZ/SearchTimeline"


# --- patch get_indices ----------------------------------------------------------
def test_get_indices_finds_ondemand_via_webpack_maps():
    # home punya map id→nama (59924:"ondemand.s") + map id→hash (59924:"abc123")
    html = 'p 59924:"ondemand.s" q 59924:"abc123" r'
    ondemand_js = 'foo(a[20],16),(b[36],16),(c[9],16)bar'

    class _FakeSession:
        async def request(self, method, url, headers=None):
            assert "ondemand.s.abc123a.js" in url  # URL dibangun dari hash
            return types.SimpleNamespace(text=ondemand_js)

    row, rest = asyncio.run(xtp._patched_get_indices(None, html, _FakeSession(), {}))
    assert row == 20
    assert rest == [36, 9]


def test_get_indices_falls_back_when_map_absent(monkeypatch):
    # tak ada map ondemand.s → harus panggil implementasi asli twikit (di-stub).
    called = {}

    async def _orig(self, home, session, headers):
        called["orig"] = True
        return 1, [2, 3]

    monkeypatch.setattr(xtp, "_original_get_indices", _orig)
    row, rest = asyncio.run(xtp._patched_get_indices(None, "tanpa map", object(), {}))
    assert called.get("orig") is True
    assert (row, rest) == (1, [2, 3])
