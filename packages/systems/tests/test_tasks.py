from __future__ import annotations

import json

from async_hbn.tasks import render_local_vllm_prompt
from async_hbn.config import ModelConfig


class CapturingTokenizer:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "rendered"


def test_local_vllm_prompt_uses_text_messages() -> None:
    tokenizer = CapturingTokenizer()
    model = ModelConfig(
        alias="qwen2_5_3b",
        family="qwen2_5",
        path="/unused/Qwen2.5-3B-Instruct",
        dtype="bfloat16", tensor_parallel_size=1, chat_template_kwargs={}, thinking_budget_tokens=None,
    )
    task = {
        "prompt": "question",
        "user_prompt": "question",
        "system_prompt": "answer format",
        "system_role": "system",
        "user_content_json": json.dumps([{"type": "text", "text": "question"}]),
    }
    assert render_local_vllm_prompt(task, model, tokenizer=tokenizer) == "rendered"
    assert tokenizer.messages == [
        {"role": "system", "content": "answer format"},
        {"role": "user", "content": "question"},
    ]
