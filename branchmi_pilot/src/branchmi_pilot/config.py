from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    kind: str = "hf"
    path: str = "HuggingFaceH4/MATH-500"
    name: str | None = None
    split: str = "test"
    question_field: str = "problem"
    answer_field: str = "answer"
    id_field: str | None = None
    revision: str | None = None
    limit: int | None = 200
    shuffle: bool = True
    seed: int = 2026


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "auto"
    device_map: str | None = "auto"
    attn_implementation: str | None = "sdpa"
    trust_remote_code: bool = True
    max_context_tokens: int | None = None


@dataclass
class PromptConfig:
    system: str = "You are a careful mathematical reasoner."
    user_template: str = (
        "Solve the following problem step by step. Put the final answer in \\boxed{{}}.\n\n"
        "Problem:\n{question}"
    )
    probe: str = (
        "This is a diagnostic probe. Based only on the reasoning shown so far, predict the "
        "most likely final answer. Do not continue the derivation. Return only "
        "<answer>FINAL_ANSWER</answer>."
    )
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": True}
    )
    probe_chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )


@dataclass
class GenerationConfig:
    baseline_max_new_tokens: int = 1536
    baseline_do_sample: bool = False
    baseline_temperature: float = 0.7
    baseline_top_p: float = 0.95
    oracle_max_total_tokens: int = 1536
    oracle_do_sample: bool = False
    oracle_temperature: float = 0.7
    oracle_top_p: float = 0.95
    candidate_top_k: int = 3
    lookahead_tokens: int = 64
    probe_samples: int = 4
    probe_max_new_tokens: int = 48
    probe_temperature: float = 0.7
    probe_top_p: float = 0.95


@dataclass
class CheckpointConfig:
    strategy: str = "hybrid"
    max_per_problem: int = 8
    stride: int = 96
    min_generated_tokens: int = 32
    tail_exclusion_tokens: int = 32
    markers: list[str] = field(
        default_factory=lambda: ["\n", "Wait", "Alternatively", "Therefore", "Thus"]
    )


@dataclass
class AnalysisConfig:
    primary_label: str = "oracle_answer_change"
    top_fraction: float = 0.10
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    min_auroc_gain: float = 0.08
    min_top_precision_relative_gain: float = 0.15
    min_probe_oracle_agreement: float = 0.70


@dataclass
class ExperimentConfig:
    run_name: str = "math500_qwen3_8b_pilot"
    output_dir: str = "outputs"
    seed: int = 2026
    resume: bool = True
    save_full_text: bool = True
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    known = {item.name for item in cls.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    section_names = {
        "dataset",
        "model",
        "prompt",
        "generation",
        "checkpoints",
        "analysis",
    }
    top_level = {key: value for key, value in raw.items() if key not in section_names}
    cfg = _construct(ExperimentConfig, top_level)
    cfg.dataset = _construct(DatasetConfig, raw.get("dataset"))
    cfg.model = _construct(ModelConfig, raw.get("model"))
    cfg.prompt = _construct(PromptConfig, raw.get("prompt"))
    cfg.generation = _construct(GenerationConfig, raw.get("generation"))
    cfg.checkpoints = _construct(CheckpointConfig, raw.get("checkpoints"))
    cfg.analysis = _construct(AnalysisConfig, raw.get("analysis"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.dataset.kind not in {"hf", "jsonl"}:
        raise ValueError("dataset.kind must be 'hf' or 'jsonl'")
    if cfg.generation.candidate_top_k < 2:
        raise ValueError("generation.candidate_top_k must be at least 2")
    if cfg.generation.probe_samples < 1:
        raise ValueError("generation.probe_samples must be positive")
    if cfg.checkpoints.strategy not in {"uniform", "stride", "markers", "hybrid"}:
        raise ValueError("Unsupported checkpoints.strategy")
    if not 0 < cfg.analysis.top_fraction <= 1:
        raise ValueError("analysis.top_fraction must be in (0, 1]")
    if cfg.analysis.primary_label not in {
        "oracle_answer_change",
        "oracle_correctness_change",
    }:
        raise ValueError("Unsupported analysis.primary_label")


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=True)
