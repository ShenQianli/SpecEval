from __future__ import annotations

import asyncio

import pytest

from async_hbn.worker import reset_async_engine


class FakeEngine:
    def __init__(self, prefix_result: bool = True) -> None:
        self.calls: list[object] = []
        self.prefix_result = prefix_result

    async def pause_generation(self, **kwargs):
        self.calls.append(("pause_generation", kwargs))

    async def reset_mm_cache(self):
        self.calls.append("reset_mm_cache")

    async def reset_encoder_cache(self):
        self.calls.append("reset_encoder_cache")

    async def reset_prefix_cache(self):
        self.calls.append("reset_prefix_cache")
        return self.prefix_result

    async def resume_generation(self):
        self.calls.append("resume_generation")

    async def check_health(self):
        self.calls.append("check_health")


def test_reset_keeps_engine_loaded_and_clears_all_request_caches() -> None:
    engine = FakeEngine()
    result = asyncio.run(reset_async_engine(engine, active_count=0))
    assert result is True
    assert engine.calls == [
        ("pause_generation", {"mode": "wait", "clear_cache": True}),
        "reset_mm_cache",
        "reset_encoder_cache",
        "reset_prefix_cache",
        "resume_generation",
        "check_health",
    ]


def test_reset_rejects_cross_run_active_requests() -> None:
    with pytest.raises(RuntimeError, match="active requests"):
        asyncio.run(reset_async_engine(FakeEngine(), active_count=1))


def test_false_prefix_reset_invalidates_reset() -> None:
    engine = FakeEngine(prefix_result=False)
    with pytest.raises(RuntimeError, match="returned false"):
        asyncio.run(reset_async_engine(engine, active_count=0))
    # finally resumes a paused worker so a subsequent health audit is possible.
    assert engine.calls[-1] == "resume_generation"

