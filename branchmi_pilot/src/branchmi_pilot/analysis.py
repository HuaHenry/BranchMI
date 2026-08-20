from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from branchmi_pilot.config import load_config
from branchmi_pilot.io_utils import deduplicate_rows, read_jsonl, write_json_atomic
from branchmi_pilot.scoring import top_fraction_precision

SCORE_FIELDS = {
    "entropy_nats": "Token entropy",
    "varentropy_nats2": "Token varentropy",
    "lookahead_js_mean_nats": "Short-lookahead JSD",
    "branchmi_weighted_nats": "BranchMI (weighted)",
    "branchmi_uniform_nats": "BranchMI (uniform)",
}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    valid = np.isfinite(scores)
    filtered_labels = labels[valid]
    filtered_scores = scores[valid]
    if len(filtered_labels) == 0 or len(np.unique(filtered_labels)) < 2:
        return None
    return float(roc_auc_score(filtered_labels, filtered_scores))


def _precision(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float | None:
    valid = np.isfinite(scores)
    if not np.any(valid):
        return None
    return top_fraction_precision(labels[valid], scores[valid], fraction)


def _interval(values: list[float], confidence_level: float) -> list[float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return None
    tail = (1.0 - confidence_level) / 2.0
    return [float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))]


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _flat_rows(rows: list[dict[str, Any]], label_field: str) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        item = {
            "problem_id": row["problem_id"],
            "checkpoint_index": row["checkpoint_index"],
            "generated_position": row["generated_position"],
            "label": int(bool(row[label_field])),
            "oracle_answer_change": int(bool(row["oracle_answer_change"])),
            "oracle_correctness_change": int(bool(row["oracle_correctness_change"])),
            "probe_oracle_branch_agreement": float(row["probe_oracle_branch_agreement"]),
        }
        for field in SCORE_FIELDS:
            item[field] = float(row[field])
        flattened.append(item)
    return flattened


def _write_checkpoint_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(flat: list[dict[str, Any]], summary: dict[str, Any], figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([row["label"] for row in flat], dtype=np.int64)
    if len(np.unique(labels)) >= 2:
        fig, axis = plt.subplots(figsize=(7.2, 5.4))
        for field, display_name in SCORE_FIELDS.items():
            scores = np.asarray([row[field] for row in flat], dtype=np.float64)
            if not np.all(np.isfinite(scores)):
                continue
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_value = summary["scores"][field]["auroc"]
            axis.plot(fpr, tpr, label=f"{display_name} ({auc_value:.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Oracle criticality ROC")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / "roc_curves.png", dpi=180)
        plt.close(fig)

    names = list(SCORE_FIELDS)
    values = [summary["scores"][name]["auroc"] for name in names]
    valid = [index for index, value in enumerate(values) if value is not None]
    if valid:
        fig, axis = plt.subplots(figsize=(8.2, 4.8))
        display_names = [SCORE_FIELDS[names[index]] for index in valid]
        plot_values = [values[index] for index in valid]
        axis.bar(display_names, plot_values, color="#4472C4")
        axis.axhline(0.5, linestyle="--", color="0.4", linewidth=1)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("AUROC")
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / "auroc_comparison.png", dpi=180)
        plt.close(fig)

    entropy_scores = np.asarray([row["entropy_nats"] for row in flat], dtype=np.float64)
    branchmi_scores = np.asarray(
        [row["branchmi_weighted_nats"] for row in flat], dtype=np.float64
    )
    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    scatter = axis.scatter(
        entropy_scores,
        branchmi_scores,
        c=labels,
        cmap="coolwarm",
        alpha=0.65,
        s=18,
    )
    axis.set(
        xlabel="Token entropy (nats)",
        ylabel="Weighted BranchMI (nats)",
        title="Local uncertainty vs. future answer influence",
    )
    axis.grid(alpha=0.2)
    legend = axis.legend(*scatter.legend_elements(), title="Oracle critical")
    axis.add_artist(legend)
    fig.tight_layout()
    fig.savefig(figures_dir / "entropy_vs_branchmi.png", dpi=180)
    plt.close(fig)


def _write_report(summary: dict[str, Any], path: Path) -> None:
    scores = summary["scores"]
    comparison = summary["branchmi_vs_entropy"]
    decision = summary["decision"]
    lines = [
        "# BranchMI pilot report",
        "",
        f"- Problems: {summary['n_problems']}",
        f"- Checkpoints: {summary['n_checkpoints']}",
        f"- Primary oracle label: `{summary['primary_label']}`",
        f"- Critical prevalence: {summary['critical_prevalence']:.3f}",
        "",
        "## Score comparison",
        "",
        "| Score | AUROC | 95% problem-bootstrap CI | Top-fraction precision |",
        "|---|---:|---:|---:|",
    ]
    for field, display_name in SCORE_FIELDS.items():
        metric = scores[field]
        ci = metric["auroc_ci"]
        ci_text = "N/A" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            f"| {display_name} | {_fmt(metric['auroc'])} | {ci_text} | "
            f"{_fmt(metric['top_fraction_precision'])} |"
        )

    lines.extend(
        [
            "",
            "## Pre-registered go/no-go checks",
            "",
            (
                f"- AUROC gain (BranchMI − entropy): {_fmt(comparison['auroc_gain'])}; "
                f"target ≥ {decision['thresholds']['min_auroc_gain']:.3f}."
            ),
            (
                f"- Top-fraction precision relative gain: "
                f"{_fmt(comparison['top_precision_relative_gain'])}; target ≥ "
                f"{decision['thresholds']['min_top_precision_relative_gain']:.3f}."
            ),
            (
                f"- Probe/full-continuation branch agreement: "
                f"{summary['probe_oracle_branch_agreement']:.3f}; target ≥ "
                f"{decision['thresholds']['min_probe_oracle_agreement']:.3f}."
            ),
            "",
            f"**Decision: {decision['recommendation'].upper()}**",
            "",
            (
                "The decision is descriptive, not a significance claim. Confidence intervals "
                "resample problems (not individual checkpoints), so checkpoints from one problem "
                "remain clustered."
            ),
            "",
            "## Artifacts",
            "",
            "- `summary.json`: machine-readable metrics and thresholds.",
            "- `checkpoint_metrics.csv`: one flat row per evaluated checkpoint.",
            "- `figures/`: ROC, AUROC comparison, and entropy-vs-BranchMI plots.",
            "- `checkpoints.jsonl`: full nested branch/probe/oracle records.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    cfg = load_config(run_path / "resolved_config.yaml")
    rows = deduplicate_rows(
        read_jsonl(run_path / "checkpoints.jsonl"),
        ("problem_id", "generated_position"),
    )
    if not rows:
        raise ValueError(f"No checkpoint records found in {run_path}")
    flat = _flat_rows(rows, cfg.analysis.primary_label)
    _write_checkpoint_csv(flat, run_path / "checkpoint_metrics.csv")

    labels = np.asarray([row["label"] for row in flat], dtype=np.int64)
    score_arrays = {
        field: np.asarray([row[field] for row in flat], dtype=np.float64)
        for field in SCORE_FIELDS
    }
    problem_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(flat):
        problem_groups[row["problem_id"]].append(index)
    problem_ids = sorted(problem_groups)

    points: dict[str, dict[str, float | None]] = {}
    for field, values in score_arrays.items():
        points[field] = {
            "auroc": _auc(labels, values),
            "top_fraction_precision": _precision(
                labels, values, cfg.analysis.top_fraction
            ),
        }

    bootstrap_auc: dict[str, list[float]] = defaultdict(list)
    bootstrap_precision: dict[str, list[float]] = defaultdict(list)
    bootstrap_auc_delta: list[float] = []
    bootstrap_precision_delta: list[float] = []
    bootstrap_precision_relative_delta: list[float] = []
    bootstrap_probe_agreement: list[float] = []
    rng = np.random.default_rng(cfg.seed + 17)
    for _ in range(cfg.analysis.bootstrap_samples):
        sampled_problem_ids = rng.choice(problem_ids, size=len(problem_ids), replace=True)
        sampled_indices = np.asarray(
            [
                index
                for problem_id in sampled_problem_ids
                for index in problem_groups[str(problem_id)]
            ],
            dtype=np.int64,
        )
        sampled_labels = labels[sampled_indices]
        sample_auc: dict[str, float] = {}
        sample_precision: dict[str, float] = {}
        for field, values in score_arrays.items():
            auc_value = _auc(sampled_labels, values[sampled_indices])
            precision_value = _precision(
                sampled_labels,
                values[sampled_indices],
                cfg.analysis.top_fraction,
            )
            if auc_value is not None:
                bootstrap_auc[field].append(auc_value)
                sample_auc[field] = auc_value
            if precision_value is not None:
                bootstrap_precision[field].append(precision_value)
                sample_precision[field] = precision_value
        if "branchmi_weighted_nats" in sample_auc and "entropy_nats" in sample_auc:
            bootstrap_auc_delta.append(
                sample_auc["branchmi_weighted_nats"] - sample_auc["entropy_nats"]
            )
        if "branchmi_weighted_nats" in sample_precision and "entropy_nats" in sample_precision:
            delta = (
                sample_precision["branchmi_weighted_nats"] - sample_precision["entropy_nats"]
            )
            bootstrap_precision_delta.append(delta)
            if sample_precision["entropy_nats"] > 0:
                bootstrap_precision_relative_delta.append(
                    delta / sample_precision["entropy_nats"]
                )
        bootstrap_probe_agreement.append(
            float(
                np.mean(
                    [flat[index]["probe_oracle_branch_agreement"] for index in sampled_indices]
                )
            )
        )

    score_summary: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        score_summary[field] = {
            **points[field],
            "auroc_ci": _interval(
                bootstrap_auc[field], cfg.analysis.confidence_level
            ),
            "top_fraction_precision_ci": _interval(
                bootstrap_precision[field], cfg.analysis.confidence_level
            ),
        }

    entropy_auc = points["entropy_nats"]["auroc"]
    branchmi_auc = points["branchmi_weighted_nats"]["auroc"]
    auc_gain = (
        None
        if entropy_auc is None or branchmi_auc is None
        else branchmi_auc - entropy_auc
    )
    entropy_precision = points["entropy_nats"]["top_fraction_precision"]
    branchmi_precision = points["branchmi_weighted_nats"]["top_fraction_precision"]
    precision_gain = (
        None
        if entropy_precision is None or branchmi_precision is None
        else branchmi_precision - entropy_precision
    )
    precision_relative_gain = (
        None
        if precision_gain is None or not entropy_precision
        else precision_gain / entropy_precision
    )
    probe_agreement = float(
        np.mean([row["probe_oracle_branch_agreement"] for row in flat])
    )

    check_auroc = auc_gain is not None and auc_gain >= cfg.analysis.min_auroc_gain
    check_precision = (
        precision_relative_gain is not None
        and precision_relative_gain >= cfg.analysis.min_top_precision_relative_gain
    )
    check_probe = probe_agreement >= cfg.analysis.min_probe_oracle_agreement
    summary = {
        "n_problems": len(problem_ids),
        "n_checkpoints": len(flat),
        "primary_label": cfg.analysis.primary_label,
        "critical_prevalence": float(np.mean(labels)),
        "top_fraction": cfg.analysis.top_fraction,
        "bootstrap_samples": cfg.analysis.bootstrap_samples,
        "scores": score_summary,
        "branchmi_vs_entropy": {
            "auroc_gain": auc_gain,
            "auroc_gain_ci": _interval(
                bootstrap_auc_delta, cfg.analysis.confidence_level
            ),
            "top_precision_absolute_gain": precision_gain,
            "top_precision_absolute_gain_ci": _interval(
                bootstrap_precision_delta, cfg.analysis.confidence_level
            ),
            "top_precision_relative_gain": precision_relative_gain,
            "top_precision_relative_gain_ci": _interval(
                bootstrap_precision_relative_delta, cfg.analysis.confidence_level
            ),
        },
        "probe_oracle_branch_agreement": probe_agreement,
        "probe_oracle_branch_agreement_ci": _interval(
            bootstrap_probe_agreement, cfg.analysis.confidence_level
        ),
        "decision": {
            "checks": {
                "auroc_gain": check_auroc,
                "top_precision_relative_gain": check_precision,
                "probe_oracle_agreement": check_probe,
            },
            "thresholds": {
                "min_auroc_gain": cfg.analysis.min_auroc_gain,
                "min_top_precision_relative_gain": cfg.analysis.min_top_precision_relative_gain,
                "min_probe_oracle_agreement": cfg.analysis.min_probe_oracle_agreement,
            },
            "recommendation": "continue" if all([check_auroc, check_precision, check_probe]) else "stop",
        },
    }
    write_json_atomic(summary, run_path / "summary.json")
    _make_plots(flat, summary, run_path / "figures")
    _write_report(summary, run_path / "report.md")
    return summary
