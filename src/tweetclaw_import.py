"""Import TweetClaw/OpenClaw tweet export files into the app's tweet schema."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COLUMNS = (
    "full_text",
    "text",
    "tweet_text",
    "tweetText",
    "content",
    "body",
)
ID_COLUMNS = ("tweet_id", "tweetId", "id_str", "id")
USERNAME_COLUMNS = ("username", "screen_name", "author_username", "author")
CREATED_COLUMNS = ("created_at", "createdAt", "date", "timestamp")


def load_tweetclaw_export(raw: bytes, filename: str) -> pd.DataFrame:
    """Read a TweetClaw/OpenClaw export file and return rows with `full_text`.

    Supported inputs:
    - JSON array
    - JSON object with a `tweets`, `items`, `results`, or `data` list
    - JSONL/NDJSON, one tweet object per line
    - CSV with a known tweet text column
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return normalize_tweet_rows(pd.read_csv(io.BytesIO(raw)))
    if suffix in {".jsonl", ".ndjson"}:
        records = _loads_json_lines(raw)
    elif suffix == ".json":
        records = _extract_records(json.loads(raw.decode("utf-8")))
    else:
        raise ValueError("Format export TweetClaw harus .csv, .json, .jsonl, atau .ndjson.")
    return normalize_tweet_rows(pd.DataFrame([_flatten_record(r) for r in records]))


def normalize_tweet_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a `full_text` column and common tweet metadata aliases."""
    if df.empty:
        return pd.DataFrame(columns=["full_text", "created_at", "username", "tweet_id"])

    out = df.copy()
    text_col = _first_existing(out, TEXT_COLUMNS)
    if text_col is None:
        raise ValueError(
            "Export tweet tidak punya kolom teks. Butuh salah satu: "
            + ", ".join(TEXT_COLUMNS)
            + "."
        )
    out["full_text"] = out[text_col].fillna("").astype(str).str.strip()

    id_col = _first_existing(out, ID_COLUMNS)
    if id_col is not None and "tweet_id" not in out.columns:
        out["tweet_id"] = out[id_col]

    username_col = _first_existing(out, USERNAME_COLUMNS)
    if username_col is not None and "username" not in out.columns:
        out["username"] = out[username_col]

    created_col = _first_existing(out, CREATED_COLUMNS)
    if created_col is not None and "created_at" not in out.columns:
        out["created_at"] = out[created_col]

    out = out[out["full_text"] != ""].reset_index(drop=True)
    if out.empty:
        raise ValueError("Export tweet tidak berisi teks yang bisa diproses.")
    return out


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _loads_json_lines(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError("Setiap baris JSONL/NDJSON harus berupa objek tweet.")
            records.append(value)
    return records


def _extract_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return _dict_records(value)
    if isinstance(value, dict):
        for key in ("tweets", "items", "results", "data"):
            rows = value.get(key)
            if isinstance(rows, list):
                return _dict_records(rows)
        return [value]
    raise ValueError("JSON export TweetClaw harus berupa objek atau array.")


def _dict_records(rows: list[Any]) -> list[dict[str, Any]]:
    records = [row for row in rows if isinstance(row, dict)]
    if len(records) != len(rows):
        raise ValueError("Export TweetClaw hanya boleh berisi objek tweet.")
    return records


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = dict(record)
    _copy_nested(flat, record, ("user", "username"), "username")
    _copy_nested(flat, record, ("user", "screen_name"), "username")
    _copy_nested(flat, record, ("author", "username"), "username")
    _copy_nested(flat, record, ("author", "screen_name"), "username")
    _copy_nested(flat, record, ("tweet", "text"), "text")
    _copy_nested(flat, record, ("tweet", "full_text"), "full_text")
    _copy_nested(flat, record, ("tweet", "id"), "tweet_id")
    _copy_nested(flat, record, ("tweet", "created_at"), "created_at")
    return flat


def _copy_nested(
    target: dict[str, Any],
    source: dict[str, Any],
    path: tuple[str, str],
    dest: str,
) -> None:
    if dest in target:
        return
    first, second = path
    nested = source.get(first)
    if isinstance(nested, dict) and second in nested:
        target[dest] = nested[second]
