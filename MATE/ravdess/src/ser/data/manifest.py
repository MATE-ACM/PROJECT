from __future__ import annotations

import json
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read JSON Lines file (one JSON object per line).

    Robust to:
    - UTF-8 BOM at file start (common on Windows) via encoding='utf-8-sig'
    - blank lines

    Raises:
        ValueError with line number context on JSON parse errors.
    """
    items: List[Dict[str, Any]] = []

    # Key fix: utf-8-sig strips BOM if present.
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            # Extra safety: in case BOM appears in the middle due to concatenation
            if lineno == 1:
                s = s.lstrip("\ufeff")

            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                preview = s[:200].replace("\t", "\\t")
                raise ValueError(
                    f"Failed to parse JSONL: {path}\n"
                    f"Line {lineno}: {e}\n"
                    f"Content preview: {preview}"
                ) from e

            if not isinstance(obj, dict):
                raise ValueError(
                    f"JSONL line must be an object (dict). Got {type(obj)} at {path}:{lineno}"
                )

            items.append(obj)

    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    """
    Write JSON Lines without BOM (utf-8).
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
