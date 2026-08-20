from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from typing_extensions import Self


def stable_int_seed(*parts: Any, modulo: int = 2**31 - 1) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}") from exc
    return rows


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def write_json_atomic(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)


def deduplicate_rows(rows: Iterable[dict[str, Any]], key_fields: tuple[str, ...]):
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        result[tuple(row[field] for field in key_fields)] = row
    return list(result.values())
