"""Patch lokal untuk init `x-client-transaction-id` twikit 2.3.3.

KENAPA ADA FILE INI
-------------------
X mewajibkan header `x-client-transaction-id` di tiap request API. twikit modern
men-generate-nya, TAPI `ClientTransaction.get_indices` twikit 2.3.3 (rilis PyPI
terbaru & git main) gagal menemukan file `ondemand.s.<hash>a.js`: ia mencari pola
lama `"ondemand.s":"<hash>"` di HTML home, padahal X sekarang memakai map webpack
terpisah:
    id→nama : `59924:"ondemand.s"`
    id→hash : `59924:"63d2b54"`
Tanpa file itu, init transaction gagal ("Couldn't get KEY_BYTE indices") → semua
crawl 404. Sisa algoritma twikit (key, animation, generate_transaction_id) BENAR;
yang rusak HANYA penemuan URL ondemand. Patch ini meng-override `get_indices` saja.

KERAPUHAN (disengaja, diketahui)
--------------------------------
Format bundle X bisa berubah tiap deploy frontend (harian/mingguan). Kalau map
webpack berubah lagi, patch ini gagal & jatuh ke implementasi asli twikit (yang
juga gagal) → crawl 404 lagi. Saat itu: periksa ulang format map di HTML home
`https://x.com` dan sesuaikan regex di bawah. Ini konsekuensi yang sudah disepakati
saat memilih jalur "patch transaction-id lokal".
"""
from __future__ import annotations

import re

import httpx
from twikit.client import gql as _gql
from twikit.x_client_transaction import transaction as _txn

# --- Resolusi query-ID GraphQL SearchTimeline ----------------------------------
# twikit 2.3.3 hardcode query-ID lama (flaR-PUMshxFWZWPNpq4zA) yang X sudah rotasi
# → 404. Ambil ID terbaru dari bundle JS web X; fallback ke ID di bawah kalau gagal.
# _FALLBACK diverifikasi aktif saat fitur dibuat; kalau resolver gagal & X rotasi
# lagi, perbarui konstanta ini.
_FALLBACK_SEARCH_QID = "Bcw3RzK-PatNAmbnw54hFw"
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)
# Anchor: URL absolut host+path literal abs.twimg.com/.../main. + hash hex (titik
# di-escape). Tolak vendor.* (segmen beda) & host lain (https:// mengikat host).
_MAIN_JS_RE = re.compile(r"https://abs\.twimg\.com/responsive-web/client-web/main\.[0-9a-f]+\.js")
# Anchor: pasangan key JS berdampingan + operationName dipatok persis "SearchTimeline"
# dengan kutip penutup → tolak BookmarkSearchTimeline / SearchTimelineV2.
_QID_RE = re.compile(r'queryId:"([^"]+)",\s*operationName:"SearchTimeline"')
_resolved_qid: str | None = None  # cache per-proses


def resolve_search_query_id(auth_token: str, ct0: str, force: bool = False) -> str:
    """Ambil query-ID SearchTimeline aktif dari bundle JS X; fallback bila gagal."""
    global _resolved_qid
    if _resolved_qid is not None and not force:
        return _resolved_qid
    cookies = {"auth_token": auth_token, "ct0": ct0}
    headers = {"user-agent": _BROWSER_UA}
    try:
        home = httpx.get("https://x.com/home", headers=headers, cookies=cookies,
                         timeout=30, follow_redirects=True).text
        m = _MAIN_JS_RE.search(home)
        if m:
            js = httpx.get(m.group(0), headers=headers, timeout=60).text
            q = _QID_RE.search(js)
            if q:
                _resolved_qid = q.group(1)
                return _resolved_qid
    except Exception:  # noqa: BLE001 — gagal apa pun → pakai cadangan
        pass
    _resolved_qid = _FALLBACK_SEARCH_QID
    return _resolved_qid


def patch_search_endpoint(qid: str) -> None:
    """Arahkan endpoint SearchTimeline twikit ke host x.com + query-ID `qid`."""
    _gql.Endpoint.SEARCH_TIMELINE = f"https://x.com/i/api/graphql/{qid}/SearchTimeline"

# Map NAMA webpack: <chunk-id>:"ondemand.s". Anchor: id angka + nama persis (titik
# di-escape, kutip penutup tepat setelah `.s`). Tak match `"...ondemand.something"`.
_CHUNK_ID_RE = re.compile(r'(\d+):"ondemand\.s"')
# Map HASH webpack untuk id spesifik: <id>:"<hex>". Nilai hex-only otomatis melewati
# entry map-nama (`"ondemand.s"` bukan hex) & ambil entry hash. \b cegah superstring.
# INDICES_REGEX dipakai ulang dari twikit (sudah benar untuk file ondemand).
_INDICES_RE = _txn.INDICES_REGEX

_ONDEMAND_URL = "https://abs.twimg.com/responsive-web/client-web/ondemand.s.{hash}a.js"

# Simpan implementasi asli untuk fallback kalau penemuan baru gagal.
_original_get_indices = _txn.ClientTransaction.get_indices


async def _patched_get_indices(self, home_page_response, session, headers):
    """Versi get_indices yang menemukan ondemand.s lewat map webpack id→nama→hash."""
    html = str(home_page_response)
    id_match = _CHUNK_ID_RE.search(html)
    if id_match:
        chunk_id = id_match.group(1)
        # \b{id}:"<hex>" — entry hash; lewati entry nama yang bukan hex.
        hash_match = re.search(rf'\b{chunk_id}:"([0-9a-f]+)"', html)
        if hash_match:
            url = _ONDEMAND_URL.format(hash=hash_match.group(1))
            resp = await session.request(method="GET", url=url, headers=headers)
            indices = [int(m.group(2)) for m in _INDICES_RE.finditer(str(resp.text))]
            if indices:
                return indices[0], indices[1:]
    # Format tak dikenali → coba implementasi asli twikit (mungkin X balik ke pola lama).
    return await _original_get_indices(self, home_page_response, session, headers)


def apply_patch() -> None:
    """Pasang override `get_indices`. Idempoten (aman dipanggil berkali-kali)."""
    _txn.ClientTransaction.get_indices = _patched_get_indices
