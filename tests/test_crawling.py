"""Tes crawling — Client palsu async yang melayani respons GraphQL mentah.

Crawl sekarang memanggil `client.gql.search_timeline` dan mem-parse respons mentah
sendiri (lihat src/crawling.py), jadi fake meniru bentuk respons X, bukan objek Tweet.
"""
import asyncio
from datetime import date

import pandas as pd
import pytest

from src import config, crawling


# --- builder respons GraphQL mentah ---------------------------------------------
def _tweet_entry(tid: str, text: str, user: str = "akun") -> dict:
    return {
        "entryId": f"tweet-{tid}",
        "content": {"itemContent": {"tweet_results": {"result": {
            "__typename": "Tweet",
            "rest_id": tid,
            "legacy": {"full_text": text, "created_at": "Wed Jan 01 00:00:00 +0000 2025", "lang": "in"},
            "core": {"user_results": {"result": {"core": {"screen_name": user}}}},
        }}}},
    }


def _cursor_entry(value: str) -> dict:
    return {"entryId": "cursor-bottom-0", "content": {"value": value}}


def _response(entries: list) -> dict:
    return {"data": {"search_by_raw_query": {"search_timeline": {"timeline": {
        "instructions": [{"entries": entries}]}}}}}


# --- fake client async ----------------------------------------------------------
_PAGES_BY_CURSOR: dict = {}  # cursor (None untuk halaman pertama) -> respons dict


class _FakeGql:
    async def search_timeline(self, query, product, count, cursor):
        return _PAGES_BY_CURSOR[cursor], None


class _FakeClient:
    def __init__(self, language=None):
        self.language = language
        self.cookies = None
        self.gql = _FakeGql()

    def set_cookies(self, cookies, clear_cookies=False):
        self.cookies = cookies


@pytest.fixture
def patched_client(monkeypatch):
    captured = {}

    def _factory(language=None):
        c = _FakeClient(language)
        captured["client"] = c
        return c

    monkeypatch.setattr(crawling, "Client", _factory)
    monkeypatch.setattr(config, "X_AUTH_TOKEN", "tok")
    monkeypatch.setattr(config, "X_CT0", "csrf")
    # Stub resolver query-ID + patch endpoint agar tes tak menyentuh jaringan/global.
    monkeypatch.setattr(crawling.x_transaction_patch, "resolve_search_query_id",
                        lambda *a, **k: "TESTQID")
    monkeypatch.setattr(crawling.x_transaction_patch, "patch_search_endpoint",
                        lambda *a, **k: None)
    return captured


# --- happy path -----------------------------------------------------------------
def test_crawl_single_page_maps_columns(patched_client):
    global _PAGES_BY_CURSOR
    _PAGES_BY_CURSOR = {None: _response([_tweet_entry("1", "halo"), _tweet_entry("2", "dunia")])}
    df = crawling.crawl_tweets("ppn", limit=10)
    assert list(df["full_text"]) == ["halo", "dunia"]
    assert list(df["username"]) == ["akun", "akun"]
    assert set(df.columns) == {"full_text", "created_at", "lang", "username", "tweet_id"}
    assert patched_client["client"].cookies == {"auth_token": "tok", "ct0": "csrf"}


def test_crawl_paginates_until_limit(patched_client):
    global _PAGES_BY_CURSOR
    _PAGES_BY_CURSOR = {
        None: _response([_tweet_entry("1", "a"), _tweet_entry("2", "b"), _cursor_entry("c1")]),
        "c1": _response([_tweet_entry("3", "c"), _tweet_entry("4", "d"), _cursor_entry("c2")]),
    }
    df = crawling.crawl_tweets("ppn", limit=4)
    assert list(df["full_text"]) == ["a", "b", "c", "d"]


# --- edge cases -----------------------------------------------------------------
def test_crawl_truncates_at_limit(patched_client):
    global _PAGES_BY_CURSOR
    _PAGES_BY_CURSOR = {
        None: _response([_tweet_entry("1", "a"), _tweet_entry("2", "b"), _cursor_entry("c1")]),
        "c1": _response([_tweet_entry("3", "c"), _tweet_entry("4", "d"), _cursor_entry("c2")]),
    }
    df = crawling.crawl_tweets("ppn", limit=3)
    assert list(df["full_text"]) == ["a", "b", "c"]


def test_crawl_dedups_repeated_ids(patched_client):
    global _PAGES_BY_CURSOR
    _PAGES_BY_CURSOR = {
        None: _response([_tweet_entry("1", "a"), _cursor_entry("c1")]),
        # halaman 2: id "1" terulang (di-dedup) + "2" baru, tanpa cursor → berhenti.
        "c1": _response([_tweet_entry("1", "a"), _tweet_entry("2", "b")]),
    }
    df = crawling.crawl_tweets("ppn", limit=10)
    assert list(df["tweet_id"]) == ["1", "2"]


def test_crawl_empty_result_returns_empty_df(patched_client):
    global _PAGES_BY_CURSOR
    _PAGES_BY_CURSOR = {None: _response([])}
    df = crawling.crawl_tweets("ppn", limit=10)
    assert df.empty


def test_crawl_stops_when_page_has_no_new_tweets(patched_client):
    global _PAGES_BY_CURSOR
    # halaman 2 mengulang id halaman 1 (tak ada baru) → berhenti, halaman 3 ("c2")
    # sengaja TIDAK disediakan; kalau loop lanjut akan KeyError (booby trap).
    _PAGES_BY_CURSOR = {
        None: _response([_tweet_entry("1", "a"), _cursor_entry("c1")]),
        "c1": _response([_tweet_entry("1", "a"), _cursor_entry("c2")]),
    }
    df = crawling.crawl_tweets("ppn", limit=10)
    assert list(df["tweet_id"]) == ["1"]


def test_crawl_stops_when_no_cursor(patched_client):
    global _PAGES_BY_CURSOR
    # halaman penuh tweet baru TAPI tanpa cursor-bottom → tak bisa lanjut → stop.
    _PAGES_BY_CURSOR = {None: _response([_tweet_entry("1", "a"), _tweet_entry("2", "b")])}
    df = crawling.crawl_tweets("ppn", limit=10)
    assert list(df["tweet_id"]) == ["1", "2"]


# --- jalan dari dalam event loop (kondisi Streamlit) ----------------------------
def test_crawl_works_inside_running_loop(patched_client):
    # Streamlit memanggil crawl_tweets dari thread yang sudah punya event loop;
    # asyncio.run mentah akan RuntimeError. _run_async harus menanganinya.
    global _PAGES_BY_CURSOR
    _PAGES_BY_CURSOR = {None: _response([_tweet_entry("1", "a")])}

    async def _main():
        return crawling.crawl_tweets("ppn", limit=5)

    df = asyncio.run(_main())
    assert list(df["tweet_id"]) == ["1"]


# --- resilience: error jaringan/rate-limit --------------------------------------
class _BoomGql:
    """search_timeline gagal pada panggilan ke-`fail_on` (1-indexed)."""
    def __init__(self, pages, fail_on):
        self._pages = pages
        self._fail_on = fail_on
        self._n = 0

    async def search_timeline(self, query, product, count, cursor):
        self._n += 1
        if self._n == self._fail_on:
            raise RuntimeError("rate limited")
        return self._pages[self._n - 1], None


def _boom_client_factory(pages, fail_on):
    def _factory(language=None):
        c = _FakeClient(language)
        c.gql = _BoomGql(pages, fail_on)
        return c
    return _factory


def test_crawl_first_page_error_propagates(patched_client, monkeypatch):
    # gagal di panggilan pertama tanpa hasil → harus raise (cookie/rate-limit).
    monkeypatch.setattr(crawling, "Client", _boom_client_factory([], fail_on=1))
    with pytest.raises(RuntimeError, match="rate limited"):
        crawling.crawl_tweets("ppn", limit=10)


def test_crawl_later_page_error_returns_partial(patched_client, monkeypatch):
    # halaman 1 sukses (ada cursor), halaman 2 gagal → kembalikan parsial halaman 1.
    page1 = _response([_tweet_entry("1", "a"), _tweet_entry("2", "b"), _cursor_entry("c1")])
    monkeypatch.setattr(crawling, "Client", _boom_client_factory([page1], fail_on=2))
    df = crawling.crawl_tweets("ppn", limit=10)
    assert list(df["tweet_id"]) == ["1", "2"]


# --- error cases ----------------------------------------------------------------
def test_crawl_missing_ct0_raises(monkeypatch):
    monkeypatch.setattr(config, "X_AUTH_TOKEN", "tok")
    monkeypatch.setattr(config, "X_CT0", "")
    with pytest.raises(RuntimeError, match="ct0"):
        crawling.crawl_tweets("ppn", limit=10)


def test_crawl_invalid_limit_raises(monkeypatch):
    monkeypatch.setattr(config, "X_AUTH_TOKEN", "tok")
    monkeypatch.setattr(config, "X_CT0", "csrf")
    with pytest.raises(ValueError):
        crawling.crawl_tweets("ppn", limit=0)


# --- parser respons mentah ------------------------------------------------------
def test_entry_to_row_extracts_fields():
    row = crawling._entry_to_row(_tweet_entry("99", "teks", user="budi"))
    assert row == {
        "full_text": "teks", "created_at": "Wed Jan 01 00:00:00 +0000 2025",
        "lang": "in", "username": "budi", "tweet_id": "99",
    }


def test_entry_to_row_returns_none_on_malformed():
    assert crawling._entry_to_row({"entryId": "tweet-1", "content": {}}) is None


def test_entry_to_row_unwraps_visibility_results():
    inner = _tweet_entry("7", "rahasia")["content"]["itemContent"]["tweet_results"]["result"]
    wrapped = {"entryId": "tweet-7", "content": {"itemContent": {"tweet_results": {"result": {
        "__typename": "TweetWithVisibilityResults", "tweet": inner}}}}}
    row = crawling._entry_to_row(wrapped)
    assert row["full_text"] == "rahasia" and row["tweet_id"] == "7"


def test_parse_entries_picks_cursor_and_skips_non_tweets():
    entries = [
        {"entryId": "who-to-follow-0", "content": {}},  # bukan tweet → dilewati
        _tweet_entry("1", "a"),
        _cursor_entry("NEXT"),
    ]
    rows, cursor = crawling._parse_tweet_entries(_response(entries))
    assert [r["tweet_id"] for r in rows] == ["1"]
    assert cursor == "NEXT"


# --- build_search_query ---------------------------------------------------------
def test_query_with_date_objects():
    q = crawling.build_search_query("PPN 12%", date(2024, 10, 15), date(2025, 5, 1))
    assert q == "PPN 12% since:2024-10-15 until:2025-05-01"


def test_query_keyword_only_when_dates_none():
    assert crawling.build_search_query("PPN 12%") == "PPN 12%"


def test_query_only_since():
    assert crawling.build_search_query("PPN", since=date(2024, 1, 2)) == "PPN since:2024-01-02"


def test_query_accepts_string_dates_and_strips_keyword():
    q = crawling.build_search_query("  PPN  ", since="2024-10-15", until="2025-05-01")
    assert q == "PPN since:2024-10-15 until:2025-05-01"


# --- select_text_column ---------------------------------------------------------
def test_select_text_column_dedups():
    df = pd.DataFrame({"full_text": ["x", "x", "y"], "lang": ["in", "in", "in"]})
    out = crawling.select_text_column(df)
    assert list(out["full_text"]) == ["x", "y"]
    assert list(out.columns) == ["full_text"]
