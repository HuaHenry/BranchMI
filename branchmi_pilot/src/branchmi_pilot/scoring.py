from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from branchmi_pilot.config import CheckpointConfig


def entropy(distribution: Iterable[float]) -> float:
    probabilities = np.asarray(list(distribution), dtype=np.float64)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def generalized_js(
    distributions: list[dict[int, float]], weights: list[float] | np.ndarray
) -> float:
    if len(distributions) < 2:
        return 0.0
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights /= normalized_weights.sum()
    support = sorted({key for distribution in distributions for key in distribution})
    matrix = np.asarray(
        [[distribution.get(key, 0.0) for key in support] for distribution in distributions],
        dtype=np.float64,
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    mixture = np.sum(normalized_weights[:, None] * matrix, axis=0)
    return entropy(mixture) - float(
        sum(weight * entropy(row) for weight, row in zip(normalized_weights, matrix))
    )


def distributions_from_labels(
    labels_by_branch: list[list[int]],
) -> list[dict[int, float]]:
    result: list[dict[int, float]] = []
    for labels in labels_by_branch:
        counts: dict[int, float] = defaultdict(float)
        for label in labels:
            counts[label] += 1.0
        total = float(len(labels))
        result.append({key: value / total for key, value in counts.items()})
    return result


def normalized_js(js_value: float, weights: Iterable[float]) -> float:
    maximum = entropy(weights)
    return 0.0 if maximum <= 0 else float(js_value / maximum)


def _evenly_cap(positions: list[int], maximum: int) -> list[int]:
    positions = sorted(set(positions))
    if len(positions) <= maximum:
        return positions
    chosen = np.linspace(0, len(positions) - 1, maximum)
    return sorted({positions[round(index)] for index in chosen})


def select_checkpoint_positions(
    generated_ids: list[int], tokenizer, cfg: CheckpointConfig
) -> list[int]:
    """Select token positions t where generated_ids[t] will be counterfactually replaced."""
    lower = cfg.min_generated_tokens
    upper = len(generated_ids) - cfg.tail_exclusion_tokens
    if upper <= lower:
        return []

    uniform: list[int] = []
    if cfg.strategy in {"uniform", "hybrid"}:
        count = min(cfg.max_per_problem, upper - lower)
        if count > 0:
            uniform = [
                round(value)
                for value in np.linspace(lower, upper - 1, num=count, endpoint=True)
            ]

    stride: list[int] = []
    if cfg.strategy in {"stride", "hybrid"}:
        stride = list(range(lower, upper, cfg.stride))

    marker_positions: list[int] = []
    if cfg.strategy in {"markers", "hybrid"}:
        rolling = ""
        for token_index, token_id in enumerate(generated_ids[:upper]):
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            rolling = (rolling + piece)[-64:]
            if token_index + 1 >= lower and any(rolling.endswith(marker) for marker in cfg.markers):
                next_position = token_index + 1
                if next_position < upper:
                    marker_positions.append(next_position)

    if cfg.strategy == "uniform":
        positions = uniform
    elif cfg.strategy == "stride":
        positions = stride
    elif cfg.strategy == "markers":
        positions = marker_positions
    else:
        # Preserve coverage while allowing semantic markers to replace nearby uniform points.
        positions = uniform + stride + marker_positions
    return _evenly_cap([position for position in positions if lower <= position < upper], cfg.max_per_problem)


def top_fraction_precision(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    if len(labels) == 0:
        return math.nan
    count = max(1, math.ceil(len(labels) * fraction))
    selected = np.argsort(-scores, kind="stable")[:count]
    return float(np.mean(labels[selected]))
