from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from branchmi_pilot.config import ModelConfig


@dataclass(frozen=True)
class SamplingOptions:
    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class GenerationBatch:
    token_ids: list[list[int]]
    texts: list[str]


@dataclass
class LookaheadBatch(GenerationBatch):
    step_js_nats: list[float]
    mean_js_nats: float
    max_js_nats: float


class TransformersBackend:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.revision:
            tokenizer_kwargs["revision"] = cfg.revision
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, **tokenizer_kwargs)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if cfg.revision:
            load_kwargs["revision"] = cfg.revision
        if cfg.device_map:
            load_kwargs["device_map"] = cfg.device_map
        dtype = self._resolve_dtype(cfg.dtype, cfg.device)
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if cfg.attn_implementation:
            load_kwargs["attn_implementation"] = cfg.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **load_kwargs)
        if not cfg.device_map:
            self.model.to(self._resolve_device(cfg.device))
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        inferred_context = getattr(self.model.config, "max_position_embeddings", None)
        self.max_context_tokens = cfg.max_context_tokens or inferred_context
        self.eos_token_ids = self._eos_token_ids()
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _resolve_dtype(name: str, device: str) -> torch.dtype | None:
        if name == "auto":
            return None
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported model dtype: {name}")
        resolved = mapping[name]
        if device == "cpu" and resolved != torch.float32:
            return torch.float32
        return resolved

    def _eos_token_ids(self) -> set[int]:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return {int(token_id) for token_id in eos}

    def set_seed(self, seed: int) -> None:
        set_seed(seed)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _pad(self, prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts:
            raise ValueError("Cannot pad an empty prompt batch")
        maximum = max(len(prompt) for prompt in prompts)
        if maximum == 0:
            raise ValueError("Prompt token lists cannot be empty")
        padded = []
        masks = []
        pad_id = int(self.tokenizer.pad_token_id)
        for prompt in prompts:
            padding = maximum - len(prompt)
            padded.append([pad_id] * padding + prompt)
            masks.append([0] * padding + [1] * len(prompt))
        return (
            torch.tensor(padded, dtype=torch.long, device=self.input_device),
            torch.tensor(masks, dtype=torch.long, device=self.input_device),
        )

    def _effective_max_new_tokens(self, prompts: list[list[int]], requested: int) -> int:
        if self.max_context_tokens is None:
            return requested
        remaining = min(int(self.max_context_tokens) - len(prompt) for prompt in prompts)
        effective = min(requested, remaining)
        if effective <= 0:
            longest = max(len(prompt) for prompt in prompts)
            raise ValueError(
                f"Prompt length {longest} leaves no generation room in context window "
                f"{self.max_context_tokens}"
            )
        return effective

    def _generation_kwargs(self, options: SamplingOptions) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if options.do_sample:
            kwargs["temperature"] = options.temperature
            kwargs["top_p"] = options.top_p
        return kwargs

    def _trim_suffix(self, suffix: list[int]) -> list[int]:
        trimmed: list[int] = []
        for token_id in suffix:
            if token_id in self.eos_token_ids or token_id == self.tokenizer.pad_token_id:
                break
            trimmed.append(int(token_id))
        return trimmed

    @torch.inference_mode()
    def generate(self, prompts: list[list[int]], options: SamplingOptions) -> GenerationBatch:
        input_ids, attention_mask = self._pad(prompts)
        effective_options = SamplingOptions(
            max_new_tokens=self._effective_max_new_tokens(prompts, options.max_new_tokens),
            do_sample=options.do_sample,
            temperature=options.temperature,
            top_p=options.top_p,
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self._generation_kwargs(effective_options),
        )
        suffixes = outputs[:, input_ids.shape[1] :].detach().cpu().tolist()
        token_ids = [self._trim_suffix(suffix) for suffix in suffixes]
        texts = [self.decode(tokens) for tokens in token_ids]
        return GenerationBatch(token_ids=token_ids, texts=texts)

    @torch.inference_mode()
    def next_token_statistics(self, prompt: list[int], top_k: int) -> dict[str, Any]:
        input_ids, attention_mask = self._pad([prompt])
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].float()
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        token_entropy = -(probabilities * log_probabilities).sum()
        surprisal = -log_probabilities
        varentropy = (probabilities * (surprisal - token_entropy).square()).sum()

        search_k = min(logits.numel(), max(top_k * 16, 64))
        top_log_probabilities, top_ids = torch.topk(log_probabilities, k=search_k)
        candidates: list[dict[str, Any]] = []
        for log_probability, token_id in zip(
            top_log_probabilities.detach().cpu().tolist(), top_ids.detach().cpu().tolist()
        ):
            token_id = int(token_id)
            token_text = self.tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if token_id in self.special_token_ids or token_text == "":
                continue
            candidates.append(
                {
                    "token_id": token_id,
                    "token_text": token_text,
                    "log_probability": float(log_probability),
                    "probability": float(math.exp(log_probability)),
                }
            )
            if len(candidates) == top_k:
                break
        if len(candidates) < 2:
            raise RuntimeError("Fewer than two usable next-token candidates")
        log_weights = torch.tensor(
            [candidate["log_probability"] for candidate in candidates], dtype=torch.float64
        )
        weights = torch.softmax(log_weights, dim=0).tolist()
        for candidate, weight in zip(candidates, weights):
            candidate["branch_weight"] = float(weight)
        return {
            "entropy_nats": float(token_entropy.item()),
            "normalized_entropy": float(token_entropy.item() / math.log(logits.numel())),
            "varentropy_nats2": float(varentropy.item()),
            "vocab_size": int(logits.numel()),
            "candidates": candidates,
        }

    @torch.inference_mode()
    def generate_lookahead(
        self, prompts: list[list[int]], max_new_tokens: int, weights: list[float]
    ) -> LookaheadBatch:
        if len(prompts) != len(weights):
            raise ValueError("Lookahead prompts and weights must have identical lengths")
        input_ids, attention_mask = self._pad(prompts)
        options = SamplingOptions(
            max_new_tokens=self._effective_max_new_tokens(prompts, max_new_tokens),
            do_sample=False,
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            **self._generation_kwargs(options),
        )
        suffix_tensor = outputs.sequences[:, input_ids.shape[1] :]
        suffixes = suffix_tensor.detach().cpu().tolist()
        token_ids = [self._trim_suffix(suffix) for suffix in suffixes]
        texts = [self.decode(tokens) for tokens in token_ids]

        score_device = outputs.scores[0].device if outputs.scores else outputs.sequences.device
        branch_weights = torch.tensor(weights, dtype=torch.float32, device=score_device)
        branch_weights = branch_weights.clamp_min(1e-30)
        alive = torch.ones(len(prompts), dtype=torch.bool, device=score_device)
        step_js: list[float] = []
        for step_index, scores in enumerate(outputs.scores):
            active_indices = torch.nonzero(alive, as_tuple=False).flatten()
            if len(active_indices) >= 2:
                active_scores = scores.index_select(0, active_indices).float()
                log_probabilities = F.log_softmax(active_scores, dim=-1)
                probabilities = log_probabilities.exp()
                active_weights = branch_weights.index_select(0, active_indices)
                active_weights = active_weights / active_weights.sum()
                log_mixture = torch.logsumexp(
                    log_probabilities + torch.log(active_weights)[:, None], dim=0
                )
                log_ratio = log_probabilities - log_mixture[None, :]
                branch_kl = torch.where(
                    probabilities > 0,
                    probabilities * log_ratio,
                    torch.zeros_like(probabilities),
                ).sum(dim=-1)
                js_value = (active_weights * branch_kl).sum()
                step_js.append(float(js_value.item()))

            if step_index < suffix_tensor.shape[1] and self.eos_token_ids:
                generated_at_step = suffix_tensor[:, step_index].to(score_device)
                ended = torch.zeros_like(alive)
                for eos_token_id in self.eos_token_ids:
                    ended |= generated_at_step == eos_token_id
                alive &= ~ended

        return LookaheadBatch(
            token_ids=token_ids,
            texts=texts,
            step_js_nats=step_js,
            mean_js_nats=float(sum(step_js) / len(step_js)) if step_js else 0.0,
            max_js_nats=max(step_js, default=0.0),
        )
