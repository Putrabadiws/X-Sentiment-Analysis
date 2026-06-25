"""Crawling tweet via `twikit` (pure-Python) — OPSIONAL.

Pengganti `tweet-harvest` (Node.js + Playwright/Chromium). tweet-harvest mati
karena Playwright/Chromium build lama yang dipin tak bisa diunduh lagi. `twikit`
memanggil API internal X langsung lewat cookie sesi — TANPA Node, TANPA browser —
jadi crawl bisa jalan dari dalam aplikasi ini.

Butuh DUA cookie dari akun X yang sudah login (DevTools → Application → Cookies):
    - auth_token : token sesi
    - ct0        : CSRF token (X menolak request tanpa header X-Csrf-Token)
tweet-harvest dulu cuma minta auth_token karena browser-nya meng-generate ct0
otomatis; tanpa browser kita harus suplai ct0 manual.

Cookie = kredensial sesi penuh — jangan commit / share / hardcode. Diisi lewat
env var (X_AUTH_TOKEN, X_CT0) atau langsung di UI.

CATATAN VERSI: butuh twikit modern (>=2.3, Python 3.10+). twikit lama (<2.x) TIDAK
meng-generate header `x-client-transaction-id` yang sekarang DIWAJIBKAN X — tanpa
itu semua request API balas 404 (code 34). twikit modern menanganinya otomatis,
plus host x.com & query-ID GraphQL terbaru, jadi tak perlu patch manual lagi.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import date

import pandas as pd
from twikit import Client
from twikit.utils import find_dict

from . import config, x_transaction_patch

# Pasang patch get_indices twikit (lihat x_transaction_patch.py) — tanpa ini init
# x-client-transaction-id gagal & semua crawl 404. Idempoten.
x_transaction_patch.apply_patch()

_MAX_PER_PAGE = 20  # batas keras search_tweet X per panggilan


def _fmt_date(d: "date | str") -> str:
    """Format tanggal jadi `YYYY-MM-DD` (format operator since/until X)."""
    if isinstance(d, date):  # datetime.date (datetime.datetime juga subclass-nya)
        return d.strftime("%Y-%m-%d")
    return str(d).strip()


def build_search_query(
    keyword: str,
    since: "date | str | None" = None,
    until: "date | str | None" = None,
) -> str:
    """Gabung kata kunci + operator tanggal X jadi satu query string.

    Dipisah dari UI supaya bisa di-tes & dipakai ulang. `since`/`until` menerima
    datetime.date (dari st.date_input) atau string `YYYY-MM-DD`; None = operator
    tanggal dilewati.

        build_search_query("PPN 12%", date(2024,10,15), date(2025,5,1))
        -> "PPN 12% since:2024-10-15 until:2025-05-01"
    """
    parts = [keyword.strip()]
    if since is not None:
        parts.append(f"since:{_fmt_date(since)}")
    if until is not None:
        parts.append(f"until:{_fmt_date(until)}")
    return " ".join(p for p in parts if p)


async def _crawl_async(
    keyword: str, limit: int, auth: str, csrf: str, product: str, language: str
) -> list[dict]:
    """Inti crawl async.

    Memanggil `client.gql.search_timeline` (low-level) lalu mem-parse respons GraphQL
    mentah SENDIRI lewat `_parse_tweet_entries`, BUKAN `client.search_tweet`. Alasan:
    kelas Tweet/User twikit 2.3.3 melempar KeyError pada skema X terbaru (mis. X
    pindahkan `created_at`/`screen_name` user dari `legacy` ke `core`) → twikit
    menelan error itu & mengembalikan 0 tweet. Parser sendiri yang hanya mengambil 5
    field (dengan fallback) jauh lebih tahan terhadap perubahan skema X.
    """
    qid = x_transaction_patch.resolve_search_query_id(auth, csrf)
    x_transaction_patch.patch_search_endpoint(qid)

    client = Client(language)
    client.set_cookies({"auth_token": auth, "ct0": csrf})

    rows: list[dict] = []
    seen: set = set()  # dedup berdasarkan id tweet (paginasi bisa tumpang-tindih)
    cursor: str | None = None
    count = min(_MAX_PER_PAGE, limit)

    while len(rows) < limit:
        try:
            response, _ = await client.gql.search_timeline(keyword, product, count, cursor)
        except Exception:  # noqa: BLE001 — rate-limit/jaringan
            if rows:
                break  # sudah ada hasil → kembalikan parsial, jangan crash
            raise  # gagal dari awal (cookie salah / rate-limit) → surfacekan ke UI
        page_rows, next_cursor = _parse_tweet_entries(response)
        added = 0
        for row in page_rows:
            tid = row.get("tweet_id")
            if tid in seen:
                continue
            seen.add(tid)
            rows.append(row)
            added += 1
            if len(rows) >= limit:
                break
        # Stop kalau limit tercapai, tak ada tweet baru, atau cursor habis (anti loop).
        if len(rows) >= limit or added == 0 or not next_cursor:
            break
        cursor = next_cursor

    return rows


def _parse_tweet_entries(response: dict) -> "tuple[list[dict], str | None]":
    """Ekstrak baris tweet + next_cursor dari respons GraphQL search mentah."""
    rows: list[dict] = []
    next_cursor: str | None = None
    entries_lists = find_dict(response, "entries", find_one=True)
    if not entries_lists:
        return rows, None
    for entry in entries_lists[0]:
        eid = entry.get("entryId", "")
        if eid.startswith("cursor-bottom"):
            next_cursor = entry.get("content", {}).get("value")
            continue
        if not eid.startswith("tweet-"):
            continue
        row = _entry_to_row(entry)
        if row is not None:
            rows.append(row)
    return rows, next_cursor


def _entry_to_row(entry: dict) -> "dict | None":
    """Petakan satu entry timeline → baris dict. Defensif terhadap field hilang."""
    try:
        result = entry["content"]["itemContent"]["tweet_results"]["result"]
    except (KeyError, TypeError):
        return None
    # TweetWithVisibilityResults membungkus tweet asli di bawah key 'tweet'.
    if result.get("__typename") == "TweetWithVisibilityResults":
        result = result.get("tweet", result)
    legacy = result.get("legacy", {})
    # X memindah created_at/name/screen_name user dari legacy ke core (lihat _crawl_async).
    user_core = (
        result.get("core", {})
        .get("user_results", {})
        .get("result", {})
        .get("core", {})
    )
    return {
        "full_text": legacy.get("full_text"),
        "created_at": legacy.get("created_at"),
        "lang": legacy.get("lang"),
        "username": user_core.get("screen_name"),
        "tweet_id": result.get("rest_id"),
    }


def crawl_tweets(
    search_keyword: str,
    limit: int = 1000,
    token: str | None = None,
    ct0: str | None = None,
    product: str = "Latest",
    language: str = "en-US",
) -> pd.DataFrame:
    """Cari tweet via twikit, kembalikan DataFrame (kolom `full_text` + metadata).

    Wrapper sinkron di atas inti async (twikit modern async) — `app.py` & tes tetap
    memanggil ini secara sinkron. search_keyword bisa pakai operator X, mis:
        'kenaikan PPN 12% until:2025-05-01 since:2024-10-15'

    product: 'Latest' (default), 'Top', atau 'Media' — sama seperti filter di UI X.
    Paginasi otomatis sampai `limit` tercapai atau data habis.
    """
    auth = token or config.X_AUTH_TOKEN
    csrf = ct0 or config.X_CT0
    if not auth or not csrf:
        raise RuntimeError(
            "Cookie X belum lengkap. Butuh auth_token DAN ct0 (set env var "
            "X_AUTH_TOKEN & X_CT0, atau isi di UI). Ambil dari DevTools akun X "
            "yang sudah login: Application → Cookies → https://x.com."
        )
    if limit < 1:
        raise ValueError("limit harus >= 1.")

    rows = _run_async(_crawl_async(search_keyword, limit, auth, csrf, product, language))
    return pd.DataFrame(rows)


def _run_async(coro):
    """Jalankan coroutine dari konteks sinkron, aman walau ada loop berjalan.

    Streamlit (>=1.x) menjalankan skrip di thread yang SUDAH punya event loop, jadi
    `asyncio.run()` di sana langsung `RuntimeError: ... running event loop`. Kalau
    loop sedang berjalan, eksekusi coro di thread terpisah (loop sendiri). Pakai
    thread — BUKAN `loop.run_until_complete` di loop yang sama — karena loop Streamlit
    tak boleh diblokir/diambil alih.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # tak ada loop berjalan → jalur normal

    box: dict = {}

    def _worker():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001 — teruskan error ke thread pemanggil
            box["error"] = e

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def select_text_column(df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
    """Dedup berdasarkan teks & sisakan kolom teks saja (seperti datasetSelected.csv)."""
    out = df.drop_duplicates(subset=[text_col])[[text_col]].reset_index(drop=True)
    return out
