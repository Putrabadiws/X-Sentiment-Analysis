"""Analisis Sentimen X — aplikasi Streamlit.

Pipeline 1 layar, 5 tab:
    1. Crawl (opsional)  2. Preprocessing  3. Labelling  4. Training & Evaluasi  5. Prediksi

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src import config, crawling, labelling, modeling, preprocessing, storage

st.set_page_config(page_title="Analisis Sentimen X", page_icon="📊", layout="wide")
config.ensure_dirs()

# CSS global minimal & defensif: rapatkan padding atas + tab sedikit lebih besar.
# Sengaja sedikit — kalau selector internal Streamlit berubah, paling efek kosmetik
# kecil yang hilang, bukan app rusak.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
      .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
      /* Rapatkan gap atas sidebar: padatkan header (baris tombol collapse),
         nolkan padding atas isi, & buang margin atas judul pertama. */
      [data-testid="stSidebarHeader"] {
        padding: 0.25rem 1rem !important; min-height: 0 !important; height: auto !important;
      }
      [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
      section[data-testid="stSidebar"] h1 { padding-top: 0 !important; margin-top: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Selaraskan matplotlib dengan theme dark: figure/axes transparan + teks terang,
# supaya chart menyatu dengan latar gelap (lihat .streamlit/config.toml).
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "text.color": "#E6EDF3",
    "axes.labelcolor": "#E6EDF3",
    "axes.edgecolor": "#3A4250",
    "xtick.color": "#C9D1D9",
    "ytick.color": "#C9D1D9",
})


# --------------------------------------------------------------------------- #
# Helper state
# --------------------------------------------------------------------------- #
def _get(key, default=None):
    return st.session_state.get(key, default)


def _set(key, value):
    st.session_state[key] = value


def _files_status() -> dict[str, bool]:
    # Hanya lexicon yang WAJIB di-upload; kamus slang punya default bawaan.
    return {"full_lexicon.csv": config.LEXICON_PATH.exists()}


def _slang_counts() -> tuple[int, int | None]:
    """(jumlah entri default bawaan, jumlah entri tambahan atau None kalau belum ada)."""
    default_n = len(preprocessing._read_slang_csv(str(config.DEFAULT_SLANG_PATH)))
    extra_n = (
        len(preprocessing._read_slang_csv(str(config.SLANG_PATH)))
        if config.SLANG_PATH.exists() else None
    )
    return default_n, extra_n


# --------------------------------------------------------------------------- #
# Komponen visual (HTML inline-style — robust, tak menarget class internal St)
# --------------------------------------------------------------------------- #
def _render_hero() -> None:
    """Banner judul bergradien di atas konten."""
    st.markdown(
        """
        <div style="padding:1.4rem 1.7rem;border-radius:18px;
             background:linear-gradient(120deg,#0f766e 0%,#155e75 55%,#1e3a8a 100%);
             border:1px solid rgba(45,212,191,0.25);
             box-shadow:0 8px 30px rgba(0,0,0,0.35);margin-bottom:1rem;">
          <div style="font-size:1.8rem;font-weight:800;letter-spacing:-0.02em;color:#f0fdfa;">
            📊 Analisis Sentimen X
          </div>
          <div style="color:#cbd5e1;margin-top:.35rem;font-size:0.95rem;">
            Crawl&nbsp;→&nbsp;Preprocessing&nbsp;→&nbsp;Labelling&nbsp;→&nbsp;Training&nbsp;→&nbsp;Prediksi
            &nbsp;·&nbsp; tweet Bahasa Indonesia &nbsp;·&nbsp; TF-IDF&nbsp;+&nbsp;SVM
            &nbsp;·&nbsp; topik contoh: kenaikan PPN 12%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stepper() -> None:
    """Indikator progres pipeline — tiap langkah menyala kalau tahapnya sudah jalan."""
    steps = [
        ("Crawl", _get("df_raw") is not None),
        ("Preprocessing", _get("df_pre") is not None),
        ("Labelling", _get("df_label") is not None),
        ("Training", config.MODEL_PATH.exists() or _get("train_result") is not None),
        ("Prediksi", config.MODEL_PATH.exists()),
    ]
    cells = []
    for i, (label, done) in enumerate(steps, 1):
        bg = "#2DD4BF" if done else "rgba(255,255,255,0.05)"
        fg = "#06231f" if done else "#94a3b8"
        ring = "rgba(45,212,191,0.9)" if done else "rgba(255,255,255,0.15)"
        lab_color = "#e6edf3" if done else "#7d8694"
        mark = "✓" if done else str(i)
        cells.append(
            f'<div style="flex:1;text-align:center;min-width:64px;">'
            f'<div style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:32px;height:32px;border-radius:50%;background:{bg};color:{fg};'
            f'font-weight:700;border:1px solid {ring};">{mark}</div>'
            f'<div style="margin-top:.35rem;font-size:0.78rem;color:{lab_color};">{label}</div>'
            f'</div>'
        )
        if i < len(steps):
            line = "#2DD4BF" if done else "rgba(255,255,255,0.1)"
            cells.append(f'<div style="flex:0.6;height:2px;background:{line};margin-top:16px;border-radius:2px;"></div>')
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:.15rem;margin:.2rem 0 .4rem;">'
        f'{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def _badge(text: str, state: str) -> str:
    """Pill status berwarna. state ∈ {ok, warn, off}. Mengembalikan HTML span."""
    palette = {
        "ok": ("rgba(45,212,191,0.12)", "#2DD4BF"),
        "warn": ("rgba(251,191,36,0.12)", "#FBBF24"),
        "off": ("rgba(255,255,255,0.05)", "#94A3B8"),
    }
    bg, fg = palette[state]
    return (
        f'<span style="display:inline-block;padding:.18rem .6rem;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:0.78rem;line-height:1.4;'
        f'border:1px solid {fg}44;">{text}</span>'
    )


def _render_badges(badges: list[str]) -> str:
    return f'<div style="display:flex;flex-wrap:wrap;gap:.4rem;">{"".join(badges)}</div>'


def _handle_support_upload(uploaded, kind: str, label: str) -> None:
    """Simpan file pendukung hasil upload, sekali per file (anti re-save tiap rerun).

    Streamlit menjalankan ulang skrip tiap interaksi & mengembalikan objek file
    yang sama; tanpa guard signature ini file ditulis ulang + cache di-clear tiap
    rerun. Signature (nama, ukuran) cukup membedakan file baru dari rerun biasa.
    """
    if uploaded is None:
        return
    sig = (uploaded.name, uploaded.size)
    state_key = f"_support_sig_{kind}"
    if st.session_state.get(state_key) == sig:
        return  # file sama, sudah tersimpan — jangan tulis/clear-cache lagi
    try:
        storage.save_support_file(uploaded.getvalue(), kind)
        st.session_state[state_key] = sig
        st.sidebar.success(f"`{label}` tersimpan & aktif.")
    except Exception as e:  # noqa: BLE001 — tampilkan error validasi/IO ke user
        st.sidebar.error(f"Gagal simpan `{label}`: {e}")


# --------------------------------------------------------------------------- #
# Sidebar — status file pendukung
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 Analisis Sentimen X")
st.sidebar.caption("Topik contoh: kenaikan PPN 12%")
st.sidebar.subheader("Upload file pendukung")
_handle_support_upload(
    st.sidebar.file_uploader(
        "kamus slang TAMBAHAN — opsional (kolom: slang, formal)", type=["csv"], key="slang_up",
        help="Digabung di atas kamus default bawaan; entri di sini menang saat bentrok. "
             "Kosongkan untuk pakai default saja.",
    ),
    "slang", "kamus_slang.csv",
)
_handle_support_upload(
    st.sidebar.file_uploader(
        "full_lexicon.csv — wajib (kolom: word, weight)", type=["csv"], key="lex_up"
    ),
    "lexicon", "full_lexicon.csv",
)

st.sidebar.subheader("Status")
_default_n, _extra_n = _slang_counts()
_lex_ok = config.LEXICON_PATH.exists()
_model_ok = config.MODEL_PATH.exists()
_status_badges = [
    _badge(f"slang default · {_default_n}", "ok"),
    _badge(f"slang tambahan · {_extra_n}", "ok") if _extra_n is not None
    else _badge("slang tambahan · —", "off"),
    _badge("full_lexicon " + ("✓" if _lex_ok else "✗ wajib"), "ok" if _lex_ok else "warn"),
    _badge("model " + ("tersedia ✓" if _model_ok else "belum ada"), "ok" if _model_ok else "off"),
]
st.sidebar.markdown(_render_badges(_status_badges), unsafe_allow_html=True)
st.sidebar.caption("kamus slang tambahan & full_lexicon.csv tersimpan otomatis ke folder data/.")

# Header utama area konten — hero + stepper progres pipeline.
_render_hero()
_render_stepper()

tab_crawl, tab_pre, tab_label, tab_train, tab_predict = st.tabs(
    ["1️⃣ Crawl", "2️⃣ Preprocessing", "3️⃣ Labelling", "4️⃣ Training & Evaluasi", "5️⃣ Prediksi"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Crawl (opsional)
# --------------------------------------------------------------------------- #
with tab_crawl:
    st.header("Ambil Data Tweet (opsional)")
    st.markdown(
        "Crawling pakai `twikit` (pure-Python, **tanpa Node/browser**). Butuh **2 cookie** "
        "akun X yang sudah login: `auth_token` & `ct0` (DevTools → Application → Cookies → "
        "`https://x.com`). Kalau tidak mau crawl, lewati tab ini dan **upload CSV** di tab "
        "Preprocessing."
    )

    keyword = st.text_input(
        "Kata kunci pencarian (tanpa operator tanggal)",
        value="kenaikan PPN 12%",
        help="Operator since/until diisi lewat kalender di bawah, jangan ketik manual di sini.",
    )
    col1, col2 = st.columns(2)
    limit = col1.number_input("Limit tweet", min_value=10, max_value=10000, value=1000, step=10)
    product = col2.selectbox("Tipe", ["Latest", "Top", "Media"], index=0)

    use_range = st.checkbox("Batasi rentang tanggal (since/until)", value=True)
    since = until = None
    if use_range:
        col_s, col_u = st.columns(2)
        since = col_s.date_input("Sejak (since:)", value=dt.date(2024, 10, 15))
        until = col_u.date_input("Sampai (until:)", value=dt.date(2025, 5, 1))

    col3, col4 = st.columns(2)
    token = col3.text_input(
        "auth_token X", type="password", value=config.X_AUTH_TOKEN,
        help="Cookie 'auth_token' akun X yang sudah login.",
    )
    ct0 = col4.text_input(
        "ct0 X", type="password", value=config.X_CT0,
        help="Cookie 'ct0' (CSRF token) akun X yang sama.",
    )

    query = crawling.build_search_query(keyword, since, until)
    st.caption(f"Query yang dikirim: `{query}`")

    if st.button("🚀 Mulai crawl", disabled=not (token and ct0)):
        if use_range and since > until:
            st.error("Tanggal 'since' harus lebih awal atau sama dengan 'until'.")
        else:
            try:
                with st.spinner("Crawling… (bisa beberapa menit)"):
                    raw = crawling.crawl_tweets(
                        query, limit=int(limit), token=token or None,
                        ct0=ct0 or None, product=product,
                    )
                    selected = crawling.select_text_column(raw)
                _set("df_raw", selected)
                st.success(f"Berhasil ambil {len(raw)} tweet, {len(selected)} unik.")
                st.dataframe(selected.head(20), use_container_width=True)
            except Exception as e:  # noqa: BLE001 — tampilkan error apa pun ke user
                st.error(f"Gagal crawl: {e}")


# --------------------------------------------------------------------------- #
# Tab 2 — Preprocessing
# --------------------------------------------------------------------------- #
with tab_pre:
    st.header("Preprocessing")
    st.markdown(
        "Tahap: cleaning → tokenizing → normalisasi slang → stopword → stemming. "
        "Kamus slang pakai **default bawaan** (+ tambahan dari sidebar bila ada)."
    )
    uploaded = st.file_uploader("Upload CSV tweet (atau pakai hasil crawl)", type=["csv"])
    df_in = None
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
    elif _get("df_raw") is not None:
        df_in = _get("df_raw")
        st.info("Memakai hasil crawl dari tab 1.")

    if df_in is not None:
        text_col = st.selectbox("Kolom teks", options=list(df_in.columns),
                                index=list(df_in.columns).index("full_text") if "full_text" in df_in.columns else 0)
        if st.button("⚙️ Jalankan preprocessing"):
            # Kamus slang default selalu ada → tak perlu guard wajib-upload lagi.
            with st.spinner("Memproses… (stemming Sastrawi bisa lama)"):
                df_pre = preprocessing.preprocess_dataframe(df_in, text_col=text_col)
            _set("df_pre", df_pre)
            st.success(f"Selesai memproses {len(df_pre)} baris.")
            st.dataframe(df_pre[["data_cleaned", "data_stemmed"]].head(20), use_container_width=True)
    else:
        st.warning("Belum ada data. Crawl dulu atau upload CSV.")


# --------------------------------------------------------------------------- #
# Tab 3 — Labelling
# --------------------------------------------------------------------------- #
with tab_label:
    st.header("Labelling (lexicon-based)")
    st.markdown("Skor = jumlah bobot kata dari `full_lexicon.csv`. >0 positive, <0 negative, =0 neutral.")
    df_pre = _get("df_pre")
    if df_pre is None:
        st.warning("Jalankan preprocessing dulu (tab 2).")
    elif not config.LEXICON_PATH.exists():
        st.error(f"`full_lexicon.csv` tidak ada di {config.DATA_DIR}.")
    else:
        if st.button("🏷️ Beri label"):
            with st.spinner("Memberi label…"):
                df_lab = labelling.label_dataframe(df_pre)
            _set("df_label", df_lab)
            st.success("Selesai labelling.")
        df_lab = _get("df_label")
        if df_lab is not None:
            counts = df_lab["sentiment"].value_counts()
            with st.container(border=True):
                st.markdown("##### 📊 Distribusi Sentimen")
                col1, col2 = st.columns([1, 2])
                col1.dataframe(counts.rename("jumlah"), use_container_width=True)
                fig, ax = plt.subplots(figsize=(4.5, 3))
                sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                            palette="viridis", legend=False)
                ax.set_xlabel("sentimen"); ax.set_ylabel("jumlah")
                fig.tight_layout()
                col2.pyplot(fig, use_container_width=False)
            st.dataframe(df_lab[["data_stemmed", "sentiment"]].head(20), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 4 — Training & Evaluasi
# --------------------------------------------------------------------------- #
with tab_train:
    st.header("Training & Evaluasi (TF-IDF + SVM + SMOTE)")
    df_label = _get("df_label")
    src_choice = st.radio(
        "Sumber data berlabel", ["Dari tab Labelling", "Upload CSV berlabel"], horizontal=True
    )
    df_train = None
    if src_choice == "Upload CSV berlabel":
        up = st.file_uploader("CSV dengan kolom data_stemmed + sentiment", type=["csv"], key="train_up")
        if up is not None:
            df_train = pd.read_csv(up)
    else:
        df_train = df_label

    if df_train is not None and {"data_stemmed", "sentiment"}.issubset(df_train.columns):
        if st.button("🤖 Latih model"):
            with st.spinner("Melatih SVM…"):
                result = modeling.train_model(df_train)
                modeling.save_model(result.pipeline)
            _set("train_result", result)
            st.success(f"Model dilatih & disimpan ke `{config.MODEL_PATH.name}`.")

        result = _get("train_result")
        if result is not None:
            with st.container(border=True):
                st.markdown("##### 📈 Metrik (held-out test set)")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Akurasi", f"{result.accuracy:.2%}")
                c2.metric("Precision", f"{result.precision:.2%}")
                c3.metric("Recall", f"{result.recall:.2%}")
                c4.metric("F1", f"{result.f1:.2%}")
                c5.metric("CV akurasi", f"{result.cv_accuracy:.2%}")
                st.caption(f"train={result.n_train} · test={result.n_test} · distribusi: {result.class_distribution}")

            col_cm, col_rep = st.columns([1, 1])
            with col_cm.container(border=True):
                st.markdown("##### 🔢 Confusion Matrix")
                # figsize kecil + use_container_width=False supaya chart tak melar selebar kolom.
                fig, ax = plt.subplots(figsize=(3.2, 2.6))
                sns.heatmap(result.confusion, annot=True, fmt="d", cmap="Blues",
                            xticklabels=result.labels, yticklabels=result.labels, ax=ax,
                            annot_kws={"size": 8})
                ax.set_xlabel("Prediksi", fontsize=8); ax.set_ylabel("Aktual", fontsize=8)
                ax.tick_params(labelsize=7)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
            with col_rep.container(border=True):
                st.markdown("##### 📋 Classification Report")
                st.code(result.report, language="text")
    else:
        st.warning("Butuh data berlabel dengan kolom `data_stemmed` & `sentiment`.")


# --------------------------------------------------------------------------- #
# Tab 5 — Prediksi
# --------------------------------------------------------------------------- #
with tab_predict:
    st.header("Prediksi Sentimen")
    if not config.MODEL_PATH.exists():
        st.warning("Belum ada model. Latih dulu di tab 4.")
    else:
        # Kamus slang default selalu ada untuk preprocessing input.
        text = st.text_area("Masukkan teks tweet", value="Kenaikan PPN 12% ini sangat memberatkan rakyat kecil")
        if st.button("🔮 Prediksi"):
            model = modeling.load_model()
            slang_dict = preprocessing.load_slang_dict()
            stop_words = preprocessing.load_stopwords()
            processed = preprocessing.preprocess_text(text, slang_dict, stop_words)
            pred = modeling.predict(model, [processed])[0]
            emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(pred, "")
            st.markdown(f"### Hasil: **{pred}** {emoji}")
            st.caption(f"Teks ter-preprocessing: `{processed}`")
