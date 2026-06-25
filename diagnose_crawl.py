"""Diagnostik crawl DI LUAR Streamlit — skrip throwaway (bukan bagian aplikasi).

Tujuan: buktikan apakah cookie-mu bisa crawl via twikit modern, tanpa gangguan
cache modul Streamlit.

Pakai (isi cookie-mu sendiri):

    X_AUTH_TOKEN="paste_auth_token" X_CT0="paste_ct0" \
        .venv/bin/python diagnose_crawl.py

Lalu tempel SELURUH output ke chat.
"""
import twikit

from src import config, crawling

print("=" * 60)
print("twikit version     :", getattr(twikit, "__version__", "n/a"))
print("auth_token terisi? :", bool(config.X_AUTH_TOKEN))
print("ct0 terisi?        :", bool(config.X_CT0))
print("=" * 60)

try:
    df = crawling.crawl_tweets("indonesia", limit=5, product="Latest")
    print(f"BERHASIL — dapat {len(df)} tweet")
    print(df[["full_text", "username"]].head(5).to_string())
except Exception:  # noqa: BLE001 — diagnostik: tampilkan traceback penuh
    import traceback

    print("GAGAL — traceback lengkap:")
    traceback.print_exc()
