"""Small shared helpers used by the ingestion, RAG, and UI layers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clean_inline_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    if any(marker in text for marker in ("\u00c3", "\u00e2", "\u0080", "\u00c2")):
        try:
            text = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00c2": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def clean_multiline_text(text: str) -> str:
    lines = [clean_inline_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def page_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "home"
    path = path.replace(".html", "")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower() or "page"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}.txt"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
