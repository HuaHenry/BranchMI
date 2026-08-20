from __future__ import annotations

from typing import Any

from branchmi_pilot.config import PromptConfig


class PromptBuilder:
    def __init__(self, tokenizer: Any, cfg: PromptConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def _messages(self, question: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.cfg.system.strip():
            messages.append({"role": "system", "content": self.cfg.system.strip()})
        messages.append(
            {
                "role": "user",
                "content": self.cfg.user_template.format(question=question),
            }
        )
        return messages

    def problem_prompt_ids(self, question: str) -> list[int]:
        messages = self._messages(question)
        if getattr(self.tokenizer, "chat_template", None):
            return list(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **self.cfg.chat_template_kwargs,
                )
            )
        rendered = "\n\n".join(
            f"{message['role'].capitalize()}: {message['content']}" for message in messages
        )
        rendered += "\n\nAssistant:"
        return list(self.tokenizer.encode(rendered, add_special_tokens=True))

    def probe_prompt_ids(self, question: str, partial_reasoning: str) -> list[int]:
        messages = self._messages(question)
        messages.extend(
            [
                {"role": "assistant", "content": partial_reasoning},
                {"role": "user", "content": self.cfg.probe},
            ]
        )
        if getattr(self.tokenizer, "chat_template", None):
            return list(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **self.cfg.probe_chat_template_kwargs,
                )
            )
        rendered = "\n\n".join(
            f"{message['role'].capitalize()}: {message['content']}" for message in messages
        )
        rendered += "\n\nAssistant:"
        return list(self.tokenizer.encode(rendered, add_special_tokens=True))

