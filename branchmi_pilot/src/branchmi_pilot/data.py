from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from branchmi_pilot.config import DatasetConfig


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    gold_answer: str | None
    metadata: dict[str, Any]


def _row_to_problem(row: dict[str, Any], index: int, cfg: DatasetConfig) -> Problem:
    if cfg.question_field not in row:
        raise KeyError(f"Question field {cfg.question_field!r} is absent from dataset row")
    question = str(row[cfg.question_field]).strip()
    answer_value = row.get(cfg.answer_field)
    answer = None if answer_value is None else str(answer_value).strip()
    raw_id = row.get(cfg.id_field) if cfg.id_field else None
    problem_id = str(raw_id) if raw_id is not None else f"{cfg.split}-{index:05d}"
    excluded = {cfg.question_field, cfg.answer_field}
    if cfg.id_field:
        excluded.add(cfg.id_field)
    metadata = {
        key: value
        for key, value in row.items()
        if key not in excluded and isinstance(value, (str, int, float, bool, type(None)))
    }
    return Problem(problem_id=problem_id, question=question, gold_answer=answer, metadata=metadata)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
    return rows


def iter_problems(cfg: DatasetConfig) -> Iterator[Problem]:
    if cfg.kind == "hf":
        kwargs: dict[str, Any] = {"split": cfg.split}
        if cfg.name is not None:
            kwargs["name"] = cfg.name
        if cfg.revision is not None:
            kwargs["revision"] = cfg.revision
        dataset = load_dataset(cfg.path, **kwargs)
        indices = np.arange(len(dataset))
        if cfg.shuffle:
            np.random.default_rng(cfg.seed).shuffle(indices)
        if cfg.limit is not None:
            indices = indices[: cfg.limit]
        for index in indices.tolist():
            yield _row_to_problem(dict(dataset[int(index)]), int(index), cfg)
        return

    rows = _load_jsonl(cfg.path)
    indices = np.arange(len(rows))
    if cfg.shuffle:
        np.random.default_rng(cfg.seed).shuffle(indices)
    if cfg.limit is not None:
        indices = indices[: cfg.limit]
    for index in indices.tolist():
        yield _row_to_problem(rows[int(index)], int(index), cfg)

