from __future__ import annotations

from async_hbn.identities import Phase, RequestKey
from async_hbn.ledger import RequestLedger, RequestRecord, RequestState


def _record(ordinal: int) -> RequestRecord:
    key = RequestKey("60", Phase.CONTINUATION, ordinal)
    return RequestRecord(key, key.logical_id("p"), seed=ordinal)


def test_ledger_distinguishes_accepted_discarded_and_aborted_cost() -> None:
    ledger = RequestLedger([_record(1), _record(2), _record(3)])
    for ordinal in (1, 2):
        key = RequestKey("60", Phase.CONTINUATION, ordinal)
        ledger.mark_submitted(
            key,
            physical_request_id=f"physical-{ordinal}",
            attempt=0,
            engine_id=0,
            timestamp_ns=1,
            prefetched=True,
        )
        ledger.mark_started(key, 2)
        ledger.mark_completed(
            key,
            completed_ns=3,
            reward_ready_ns=4,
            reward=0,
            raw_output="x",
            prompt_tokens=10,
            completion_tokens=ordinal,
            finish_reason="stop",
        )
    key3 = RequestKey("60", Phase.CONTINUATION, 3)
    ledger.mark_submitted(
        key3,
        physical_request_id="physical-3",
        attempt=0,
        engine_id=0,
        timestamp_ns=1,
        prefetched=True,
    )
    ledger.mark_started(key3, 2)
    ledger[key3].prompt_tokens_before_abort = 10
    ledger[key3].generated_tokens_before_abort = 7
    ledger.resolve_selection(frozenset({RequestKey("60", Phase.CONTINUATION, 1)}))
    assert ledger[RequestKey("60", Phase.CONTINUATION, 1)].state is RequestState.ACCEPTED
    assert ledger[RequestKey("60", Phase.CONTINUATION, 2)].state is RequestState.COMPLETED_DISCARDED
    assert ledger[key3].state is RequestState.ABORTED_RUNNING
    costs = ledger.token_accounting()
    assert costs["accepted_completion_tokens"] == 1
    assert costs["discarded_completion_tokens"] == 2
    assert costs["aborted_prompt_tokens"] == 10
    assert costs["aborted_generated_tokens"] == 7
