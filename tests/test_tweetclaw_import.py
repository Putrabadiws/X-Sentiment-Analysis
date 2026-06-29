"""Tes import export TweetClaw/OpenClaw lokal."""
import json

import pandas as pd
import pytest

from src import tweetclaw_import


def test_load_json_object_with_nested_tweets():
    raw = json.dumps(
        {
            "tweets": [
                {
                    "text": "PPN 12% bikin berat",
                    "id": "1",
                    "author": {"username": "warga"},
                    "createdAt": "2026-01-01T00:00:00Z",
                }
            ]
        }
    ).encode()
    df = tweetclaw_import.load_tweetclaw_export(raw, "tweetclaw.json")
    assert list(df["full_text"]) == ["PPN 12% bikin berat"]
    assert list(df["tweet_id"]) == ["1"]
    assert list(df["username"]) == ["warga"]
    assert list(df["created_at"]) == ["2026-01-01T00:00:00Z"]


def test_load_jsonl_skips_blank_lines_and_drops_blank_text():
    raw = (
        b'{"tweet": {"text": "bagus sekali", "id": "7"}}\n'
        b"\n"
        b'{"text": "   "}\n'
    )
    df = tweetclaw_import.load_tweetclaw_export(raw, "tweetclaw.jsonl")
    assert list(df["full_text"]) == ["bagus sekali"]
    assert list(df["tweet_id"]) == ["7"]


def test_normalize_csv_style_rows_adds_full_text_alias():
    source = pd.DataFrame({"tweetText": ["naik turun"], "screen_name": ["akun"]})
    df = tweetclaw_import.normalize_tweet_rows(source)
    assert list(df["full_text"]) == ["naik turun"]
    assert list(df["username"]) == ["akun"]
    assert "tweetText" in df.columns


def test_invalid_export_without_text_column_raises():
    raw = json.dumps({"tweets": [{"id": "1"}]}).encode()
    with pytest.raises(ValueError, match="kolom teks"):
        tweetclaw_import.load_tweetclaw_export(raw, "tweetclaw.json")
