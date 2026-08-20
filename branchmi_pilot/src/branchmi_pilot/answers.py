from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

NO_ANSWER = "<NO_ANSWER>"


def _balanced_braced(text: str, opening_brace: int) -> str | None:
    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace + 1 : index]
    return None


def extract_final_answer(text: str) -> str:
    if not text or not text.strip():
        return NO_ANSWER

    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged[-1].strip() or NO_ANSWER

    boxed_positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(boxed_positions):
        brace = text.find("{", start)
        boxed = _balanced_braced(text, brace)
        if boxed and boxed.strip():
            return boxed.strip()

    hashes = re.findall(r"####\s*(.+)", text)
    if hashes:
        return hashes[-1].strip()

    finals = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if finals:
        return finals[-1].strip().strip("$*` ")

    display_math = re.findall(r"\\\[(.*?)\\\]", text, flags=re.DOTALL)
    if display_math:
        return display_math[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return NO_ANSWER
    candidate = re.sub(r"^(?:therefore|thus|so)[,:]?\s*", "", lines[-1], flags=re.IGNORECASE)
    return candidate.strip().strip("$*` ") or NO_ANSWER


def canonicalize_answer(answer: str) -> str:
    if not answer or answer == NO_ANSWER:
        return NO_ANSWER
    value = answer.strip().lower()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "")
    value = value.replace("\u2212", "-")
    value = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    value = value.strip(".$")
    return value or NO_ANSWER


@dataclass
class AnswerClusterResult:
    labels: list[int]
    representatives: list[str]


class AnswerEvaluator:
    """Answer extraction, symbolic equivalence, correctness, and clustering."""

    def __init__(self):
        self._parse_cache: dict[str, Any] = {}
        try:
            from math_verify import parse, verify

            self._parse_fn = parse
            self._verify_fn = verify
            self.math_verify_available = True
        except ImportError:
            self._parse_fn = None
            self._verify_fn = None
            self.math_verify_available = False

    def _parse(self, answer: str):
        if answer in self._parse_cache:
            return self._parse_cache[answer]
        if not self.math_verify_available or answer == NO_ANSWER:
            parsed = []
        else:
            wrapped = answer if "$" in answer else f"${answer}$"
            try:
                parsed = self._parse_fn(wrapped, raise_on_error=False)
            except (ValueError, TypeError, TimeoutError):
                parsed = []
        self._parse_cache[answer] = parsed
        return parsed

    def equivalent(self, first: str, second: str) -> bool:
        if canonicalize_answer(first) == canonicalize_answer(second):
            return True
        if first == NO_ANSWER or second == NO_ANSWER or not self.math_verify_available:
            return False
        parsed_first = self._parse(first)
        parsed_second = self._parse(second)
        if not parsed_first or not parsed_second:
            return False
        try:
            return bool(self._verify_fn(parsed_first, parsed_second, raise_on_error=False))
        except (ValueError, TypeError, TimeoutError):
            return False

    def response_is_correct(self, response: str, gold_answer: str | None) -> bool | None:
        if gold_answer is None:
            return None
        prediction = extract_final_answer(response)
        return self.equivalent(gold_answer, prediction)

    def cluster(self, answers: list[str]) -> AnswerClusterResult:
        representatives: list[str] = []
        labels: list[int] = []
        for answer in answers:
            assigned = None
            for label, representative in enumerate(representatives):
                if self.equivalent(answer, representative):
                    assigned = label
                    break
            if assigned is None:
                assigned = len(representatives)
                representatives.append(answer)
            labels.append(assigned)
        return AnswerClusterResult(labels=labels, representatives=representatives)

