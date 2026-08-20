from __future__ import annotations

import platform
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from branchmi_pilot.answers import AnswerEvaluator, extract_final_answer
from branchmi_pilot.backend import SamplingOptions, TransformersBackend
from branchmi_pilot.config import ExperimentConfig, save_config
from branchmi_pilot.data import Problem, iter_problems
from branchmi_pilot.io_utils import JsonlWriter, read_jsonl, stable_int_seed, write_json_atomic
from branchmi_pilot.prompting import PromptBuilder
from branchmi_pilot.scoring import (
    distributions_from_labels,
    generalized_js,
    normalized_js,
    select_checkpoint_positions,
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _mode_label(labels: list[int]) -> int:
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


class PilotExperiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_run_directory()
        self.backend = TransformersBackend(cfg.model)
        self.prompts = PromptBuilder(self.backend.tokenizer, cfg.prompt)
        self.answers = AnswerEvaluator()

    def _prepare_run_directory(self) -> None:
        resolved_path = self.run_dir / "resolved_config.yaml"
        if resolved_path.exists():
            with resolved_path.open("r", encoding="utf-8") as handle:
                existing = yaml.safe_load(handle)
            if existing != self.cfg.to_dict():
                raise ValueError(
                    f"Run directory {self.run_dir} contains a different configuration. "
                    "Use another run_name, or restore the original config before resuming."
                )
            if not self.cfg.resume:
                raise FileExistsError(
                    f"Run directory {self.run_dir} already exists and resume=false"
                )
        else:
            save_config(self.cfg, resolved_path)

        metadata = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "packages": {
                name: _package_version(name)
                for name in [
                    "transformers",
                    "datasets",
                    "accelerate",
                    "math-verify",
                    "scikit-learn",
                ]
            },
        }
        write_json_atomic(metadata, self.run_dir / "environment.json")

    def _baseline_record(self, problem: Problem) -> dict[str, Any]:
        prompt_ids = self.prompts.problem_prompt_ids(problem.question)
        self.backend.set_seed(stable_int_seed(self.cfg.seed, problem.problem_id, "baseline"))
        options = SamplingOptions(
            max_new_tokens=self.cfg.generation.baseline_max_new_tokens,
            do_sample=self.cfg.generation.baseline_do_sample,
            temperature=self.cfg.generation.baseline_temperature,
            top_p=self.cfg.generation.baseline_top_p,
        )
        generated = self.backend.generate([prompt_ids], options)
        response = generated.texts[0]
        record: dict[str, Any] = {
            "problem_id": problem.problem_id,
            "question": problem.question,
            "gold_answer": problem.gold_answer,
            "metadata": problem.metadata,
            "prompt_token_count": len(prompt_ids),
            "generated_token_ids": generated.token_ids[0],
            "generated_token_count": len(generated.token_ids[0]),
            "baseline_answer": extract_final_answer(response),
            "baseline_correct": self.answers.response_is_correct(response, problem.gold_answer),
        }
        if self.cfg.save_full_text:
            record["baseline_response"] = response
        return record

    def _probe_distributions(
        self,
        problem: Problem,
        partial_reasoning_by_branch: list[str],
        checkpoint_position: int,
    ) -> tuple[list[list[str]], list[list[int]], list[str], list[dict[int, float]]]:
        branch_count = len(partial_reasoning_by_branch)
        sample_count = self.cfg.generation.probe_samples
        probe_prompts: list[list[int]] = []
        for partial_reasoning in partial_reasoning_by_branch:
            prompt_ids = self.prompts.probe_prompt_ids(problem.question, partial_reasoning)
            probe_prompts.extend([prompt_ids] * sample_count)

        self.backend.set_seed(
            stable_int_seed(self.cfg.seed, problem.problem_id, checkpoint_position, "probe")
        )
        generated = self.backend.generate(
            probe_prompts,
            SamplingOptions(
                max_new_tokens=self.cfg.generation.probe_max_new_tokens,
                do_sample=True,
                temperature=self.cfg.generation.probe_temperature,
                top_p=self.cfg.generation.probe_top_p,
            ),
        )
        completion_groups = [
            generated.texts[index * sample_count : (index + 1) * sample_count]
            for index in range(branch_count)
        ]
        answer_groups = [
            [extract_final_answer(completion) for completion in completions]
            for completions in completion_groups
        ]
        flat_answers = [answer for group in answer_groups for answer in group]
        clustering = self.answers.cluster(flat_answers)
        label_groups = [
            clustering.labels[index * sample_count : (index + 1) * sample_count]
            for index in range(branch_count)
        ]
        distributions = distributions_from_labels(label_groups)
        representatives = clustering.representatives
        return completion_groups, label_groups, representatives, distributions

    def _checkpoint_record(
        self,
        problem: Problem,
        baseline: dict[str, Any],
        position: int,
        checkpoint_index: int,
    ) -> dict[str, Any]:
        generated_ids = [int(token_id) for token_id in baseline["generated_token_ids"]]
        problem_prompt_ids = self.prompts.problem_prompt_ids(problem.question)
        model_prefix = problem_prompt_ids + generated_ids[:position]
        token_statistics = self.backend.next_token_statistics(
            model_prefix, self.cfg.generation.candidate_top_k
        )
        candidates = token_statistics.pop("candidates")
        candidate_ids = [int(candidate["token_id"]) for candidate in candidates]
        weights = [float(candidate["branch_weight"]) for candidate in candidates]
        candidate_prompts = [model_prefix + [token_id] for token_id in candidate_ids]

        lookahead = self.backend.generate_lookahead(
            candidate_prompts,
            max_new_tokens=self.cfg.generation.lookahead_tokens,
            weights=weights,
        )
        partial_reasoning_ids = [
            generated_ids[:position] + [candidate_id] + continuation
            for candidate_id, continuation in zip(candidate_ids, lookahead.token_ids)
        ]
        partial_reasoning = [self.backend.decode(tokens) for tokens in partial_reasoning_ids]

        probe_completions, probe_label_groups, probe_representatives, probe_distributions = (
            self._probe_distributions(problem, partial_reasoning, position)
        )
        branchmi_weighted = generalized_js(probe_distributions, weights)
        uniform_weights = [1.0 / len(weights)] * len(weights)
        branchmi_uniform = generalized_js(probe_distributions, uniform_weights)
        probe_mode_labels = [_mode_label(labels) for labels in probe_label_groups]
        probe_mode_answers = [probe_representatives[label] for label in probe_mode_labels]

        oracle_remaining = max(
            1, self.cfg.generation.oracle_max_total_tokens - (position + 1)
        )
        self.backend.set_seed(
            stable_int_seed(self.cfg.seed, problem.problem_id, position, "oracle")
        )
        oracle_generated = self.backend.generate(
            candidate_prompts,
            SamplingOptions(
                max_new_tokens=oracle_remaining,
                do_sample=self.cfg.generation.oracle_do_sample,
                temperature=self.cfg.generation.oracle_temperature,
                top_p=self.cfg.generation.oracle_top_p,
            ),
        )
        full_reasoning_ids = [
            generated_ids[:position] + [candidate_id] + continuation
            for candidate_id, continuation in zip(candidate_ids, oracle_generated.token_ids)
        ]
        full_responses = [self.backend.decode(tokens) for tokens in full_reasoning_ids]
        oracle_answers = [extract_final_answer(response) for response in full_responses]
        oracle_clustering = self.answers.cluster(oracle_answers)
        oracle_correct = [
            self.answers.response_is_correct(response, problem.gold_answer)
            for response in full_responses
        ]
        correctness_values = {value for value in oracle_correct if value is not None}
        probe_matches_oracle = [
            self.answers.equivalent(probe_answer, oracle_answer)
            for probe_answer, oracle_answer in zip(probe_mode_answers, oracle_answers)
        ]

        branches: list[dict[str, Any]] = []
        for branch_index, candidate in enumerate(candidates):
            distribution = {
                str(label): probability
                for label, probability in probe_distributions[branch_index].items()
            }
            branch: dict[str, Any] = {
                **candidate,
                "lookahead_token_count": len(lookahead.token_ids[branch_index]),
                "probe_answers": [
                    extract_final_answer(text) for text in probe_completions[branch_index]
                ],
                "probe_answer_cluster_distribution": distribution,
                "probe_mode_answer": probe_mode_answers[branch_index],
                "oracle_answer": oracle_answers[branch_index],
                "oracle_answer_cluster": oracle_clustering.labels[branch_index],
                "oracle_correct": oracle_correct[branch_index],
                "probe_matches_oracle": probe_matches_oracle[branch_index],
                "oracle_continuation_token_count": len(
                    oracle_generated.token_ids[branch_index]
                ),
            }
            if self.cfg.save_full_text:
                branch.update(
                    {
                        "lookahead_text": lookahead.texts[branch_index],
                        "probe_completions": probe_completions[branch_index],
                        "oracle_full_response": full_responses[branch_index],
                    }
                )
            branches.append(branch)

        return {
            "problem_id": problem.problem_id,
            "checkpoint_index": checkpoint_index,
            "generated_position": position,
            "model_prefix_token_count": len(model_prefix),
            "baseline_token_id_at_position": generated_ids[position],
            "baseline_token_text_at_position": self.backend.tokenizer.decode(
                [generated_ids[position]], skip_special_tokens=False
            ),
            **token_statistics,
            "lookahead_js_mean_nats": lookahead.mean_js_nats,
            "lookahead_js_max_nats": lookahead.max_js_nats,
            "lookahead_js_by_step_nats": lookahead.step_js_nats,
            "branchmi_weighted_nats": branchmi_weighted,
            "branchmi_weighted_normalized": normalized_js(branchmi_weighted, weights),
            "branchmi_uniform_nats": branchmi_uniform,
            "branchmi_uniform_normalized": normalized_js(branchmi_uniform, uniform_weights),
            "oracle_answer_change": len(set(oracle_clustering.labels)) > 1,
            "oracle_correctness_change": len(correctness_values) > 1,
            "oracle_unique_answer_count": len(set(oracle_clustering.labels)),
            "probe_oracle_branch_agreement": float(np.mean(probe_matches_oracle)),
            "probe_oracle_all_branches_match": all(probe_matches_oracle),
            "probe_answer_cluster_representatives": probe_representatives,
            "oracle_answer_cluster_representatives": oracle_clustering.representatives,
            "branches": branches,
        }

    def run(self) -> Path:
        problem_path = self.run_dir / "problems.jsonl"
        checkpoint_path = self.run_dir / "checkpoints.jsonl"
        existing_problems = {
            row["problem_id"]: row for row in read_jsonl(problem_path)
        }
        existing_checkpoints = {
            (row["problem_id"], int(row["generated_position"]))
            for row in read_jsonl(checkpoint_path)
        }

        problems = list(iter_problems(self.cfg.dataset))
        with JsonlWriter(problem_path) as problem_writer, JsonlWriter(
            checkpoint_path
        ) as checkpoint_writer:
            for problem in tqdm(problems, desc="Problems", unit="problem"):
                baseline = existing_problems.get(problem.problem_id)
                if baseline is None:
                    baseline = self._baseline_record(problem)
                    problem_writer.write(baseline)
                    existing_problems[problem.problem_id] = baseline

                positions = select_checkpoint_positions(
                    [int(token_id) for token_id in baseline["generated_token_ids"]],
                    self.backend.tokenizer,
                    self.cfg.checkpoints,
                )
                for checkpoint_index, position in enumerate(
                    tqdm(positions, desc=problem.problem_id, leave=False, unit="checkpoint")
                ):
                    key = (problem.problem_id, position)
                    if key in existing_checkpoints:
                        continue
                    record = self._checkpoint_record(
                        problem, baseline, position, checkpoint_index
                    )
                    checkpoint_writer.write(record)
                    existing_checkpoints.add(key)

        from branchmi_pilot.analysis import analyze_run

        analyze_run(self.run_dir)
        return self.run_dir
