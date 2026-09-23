"""One persistent AsyncLLM replica per GPU.

vLLM and torch are imported only inside the spawned process, after
``CUDA_VISIBLE_DEVICES`` is fixed. The process survives all policy runs.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

from .messages import CommandType, EventType, event


@dataclass
class _ActiveRequest:
    payload: Mapping[str, Any]
    task: asyncio.Task[None] | None = None
    last_output: Any = None
    abort_requested: bool = False
    terminal_emitted: bool = False


async def reset_async_engine(engine: Any, active_count: int) -> bool:
    """Drain/clear logical and KV state while retaining loaded model state."""

    if active_count:
        raise RuntimeError(f"cannot reset with {active_count} active requests")
    paused = False
    try:
        await engine.pause_generation(mode="wait", clear_cache=True)
        paused = True
        await engine.reset_mm_cache()
        await engine.reset_encoder_cache()
        prefix_reset = bool(await engine.reset_prefix_cache())
        if not prefix_reset:
            raise RuntimeError("reset_prefix_cache returned false")
        await engine.resume_generation()
        paused = False
        await engine.check_health()
        return prefix_reset
    finally:
        if paused:
            # Best effort: leave a failed worker schedulable for health checks;
            # the router will invalidate the sample and decide whether to restart.
            try:
                await engine.resume_generation()
            except Exception:
                pass


def _token_counts(output: Any) -> tuple[int, int]:
    if output is None:
        return 0, 0
    prompt_ids = getattr(output, "prompt_token_ids", None) or []
    candidates = getattr(output, "outputs", None) or []
    candidate = candidates[0] if candidates else None
    completion_ids = [] if candidate is None else (getattr(candidate, "token_ids", None) or [])
    return len(prompt_ids), len(completion_ids)


def _final_fields(output: Any) -> dict[str, Any]:
    prompt_tokens, completion_tokens = _token_counts(output)
    candidates = getattr(output, "outputs", None) or []
    candidate = candidates[0] if candidates else None
    return {
        "raw_output": "" if candidate is None else str(getattr(candidate, "text", "") or ""),
        "finish_reason": (
            "error" if candidate is None else str(getattr(candidate, "finish_reason", None) or "stop")
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


class _WorkerRuntime:
    def __init__(
        self,
        *,
        engine_id: int,
        engine: Any,
        sampling_params_factory: Any,
        command_queue: Any,
        event_queue: Any,
    ) -> None:
        self.engine_id = int(engine_id)
        self.engine = engine
        self.sampling_params_factory = sampling_params_factory
        self.command_queue = command_queue
        self.event_queue = event_queue
        self.active: dict[str, _ActiveRequest] = {}
        self.closed = False

    def emit(self, event_type: EventType, **payload: Any) -> None:
        self.event_queue.put(event(
            event_type,
            engine_id=self.engine_id,
            timestamp_ns=time.monotonic_ns(),
            **payload,
        ))

    async def generate_one(self, payload: Mapping[str, Any]) -> None:
        request_id = str(payload["physical_request_id"])
        active = self.active[request_id]
        self.emit(
            EventType.STARTED,
            run_uuid=payload["run_uuid"],
            logical_request_id=payload["logical_request_id"],
            physical_request_id=request_id,
            attempt=int(payload["attempt"]),
        )
        try:
            params = self.sampling_params_factory(
                int(payload["seed"]), payload.get("max_new_tokens")
            )
            async for output in self.engine.generate(
                str(payload["prompt"]), params, request_id=request_id
            ):
                active.last_output = output
            fields = _final_fields(active.last_output)
            if active.abort_requested:
                self.emit(
                    EventType.ABORTED,
                    run_uuid=payload["run_uuid"],
                    logical_request_id=payload["logical_request_id"],
                    physical_request_id=request_id,
                    prompt_tokens_before_abort=fields["prompt_tokens"],
                    generated_tokens_before_abort=fields["completion_tokens"],
                )
            elif active.last_output is None:
                self.emit(
                    EventType.INFRA_ERROR,
                    run_uuid=payload["run_uuid"],
                    logical_request_id=payload["logical_request_id"],
                    physical_request_id=request_id,
                    error="generation returned no output",
                )
            else:
                self.emit(
                    EventType.COMPLETED,
                    run_uuid=payload["run_uuid"],
                    logical_request_id=payload["logical_request_id"],
                    physical_request_id=request_id,
                    **fields,
                )
            active.terminal_emitted = True
        except asyncio.CancelledError:
            prompt_tokens, completion_tokens = _token_counts(active.last_output)
            self.emit(
                EventType.ABORTED,
                run_uuid=payload["run_uuid"],
                logical_request_id=payload["logical_request_id"],
                physical_request_id=request_id,
                prompt_tokens_before_abort=prompt_tokens,
                generated_tokens_before_abort=completion_tokens,
            )
            active.terminal_emitted = True
            raise
        except BaseException as exc:
            self.emit(
                EventType.INFRA_ERROR,
                run_uuid=payload["run_uuid"],
                logical_request_id=payload["logical_request_id"],
                physical_request_id=request_id,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            active.terminal_emitted = True
        finally:
            self.active.pop(request_id, None)

    async def submit(self, payloads: list[Mapping[str, Any]]) -> None:
        for payload in payloads:
            request_id = str(payload["physical_request_id"])
            if request_id in self.active:
                raise ValueError(f"duplicate active physical request ID {request_id}")
            active = _ActiveRequest(payload=dict(payload))
            self.active[request_id] = active
            active.task = asyncio.create_task(self.generate_one(payload))

    async def abort(self, request_ids: list[str]) -> None:
        owned = [request_id for request_id in request_ids if request_id in self.active]
        if not owned:
            return
        for request_id in owned:
            self.active[request_id].abort_requested = True
        await self.engine.abort(owned)
        tasks = [
            active.task
            for request_id in owned
            if (active := self.active.get(request_id)) is not None
            and active.task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def reset(self, reset_id: str) -> None:
        start = time.monotonic_ns()
        try:
            success = await reset_async_engine(self.engine, len(self.active))
            self.emit(
                EventType.RESET_COMPLETE,
                reset_id=reset_id,
                success=success,
                reset_duration_ns=time.monotonic_ns() - start,
                active_count=len(self.active),
            )
        except BaseException as exc:
            self.emit(
                EventType.RESET_COMPLETE,
                reset_id=reset_id,
                success=False,
                reset_duration_ns=time.monotonic_ns() - start,
                active_count=len(self.active),
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )

    async def run(self) -> None:
        self.emit(EventType.READY)
        while not self.closed:
            message = await asyncio.to_thread(self.command_queue.get)
            command_type = CommandType(str(message["type"]))
            try:
                if command_type is CommandType.SUBMIT:
                    await self.submit(list(message["requests"]))
                elif command_type is CommandType.ABORT:
                    await self.abort(list(map(str, message["physical_request_ids"])))
                elif command_type is CommandType.RESET:
                    await self.reset(str(message["reset_id"]))
                elif command_type is CommandType.HEALTH:
                    await self.engine.check_health()
                    self.emit(EventType.HEALTHY, health_id=str(message["health_id"]))
                elif command_type is CommandType.SHUTDOWN:
                    if self.active:
                        await self.abort(list(self.active))
                    self.closed = True
                else:  # pragma: no cover - enum guards this
                    raise ValueError(f"unsupported command {command_type}")
            except BaseException as exc:
                self.emit(
                    EventType.WORKER_ERROR,
                    command=command_type.value,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
        self.engine.shutdown()
        self.emit(EventType.SHUTDOWN_COMPLETE)


def worker_process_main(
    engine_id: int,
    gpu_id: int,
    command_queue: Any,
    event_queue: Any,
    model_config: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if bool(engine_config.get("batch_invariant", False)):
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    else:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)

    async def main() -> None:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        kwargs = {
            "model": str(model_config["path"]),
            "tokenizer": str(model_config["path"]),
            "dtype": str(model_config["dtype"]),
            "tensor_parallel_size": int(model_config["tensor_parallel_size"]),
            "max_model_len": int(engine_config["max_model_len"]),
            "gpu_memory_utilization": float(engine_config["gpu_memory_utilization"]),
            "max_num_seqs": int(engine_config["max_num_seqs"]),
            "enable_prefix_caching": bool(engine_config["enable_prefix_caching"]),
            "trust_remote_code": True,
            "disable_log_stats": False,
        }
        reasoning_parser = engine_config.get("reasoning_parser")
        if reasoning_parser:
            kwargs["reasoning_parser"] = str(reasoning_parser)
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))

        base_sampling = {
            "temperature": float(sampling_config["temperature"]),
            "top_p": float(sampling_config["top_p"]),
            "top_k": int(sampling_config["top_k"]),
            "max_tokens": int(sampling_config["max_new_tokens"]),
        }
        thinking_budget = model_config.get("thinking_budget_tokens")
        if thinking_budget is not None:
            base_sampling["thinking_token_budget"] = int(thinking_budget)

        def sampling_params_factory(seed: int, max_new_tokens: int | None = None) -> Any:
            values = dict(base_sampling)
            if max_new_tokens is not None:
                values["max_tokens"] = int(max_new_tokens)
            return SamplingParams(**values, seed=seed)

        runtime = _WorkerRuntime(
            engine_id=engine_id,
            engine=engine,
            sampling_params_factory=sampling_params_factory,
            command_queue=command_queue,
            event_queue=event_queue,
        )
        await runtime.run()

    try:
        asyncio.run(main())
    except BaseException as exc:
        event_queue.put(event(
            EventType.WORKER_ERROR,
            engine_id=engine_id,
            timestamp_ns=time.monotonic_ns(),
            command="startup",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        ))
        raise
