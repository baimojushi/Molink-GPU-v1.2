from worker.result_builder import terminal_progress_contract
from molink_causal.fields import FieldSnapshot
from molink_causal.state_machine import Event, State, next_state


def test_render_review_is_completed_upload_review_not_failed():
    assert terminal_progress_contract("render_review") == (
        "r2_upload",
        "done",
        92,
        "效果图已生成，正在复核",
    )


def test_terminal_progress_keeps_success_and_failure_distinct():
    assert terminal_progress_contract("succeeded")[:3] == ("done", "done", 100)
    assert terminal_progress_contract("succeeded_partial")[:3] == ("done", "done", 100)
    assert terminal_progress_contract("failed")[:3] == ("failed", "failed", 92)


def test_render_review_progress_drives_review_state_not_failed_state():
    stage, state, pct, _message = terminal_progress_contract("render_review")
    snapshot = FieldSnapshot().merge({"delivery": {"partial_review_count": 1, "final_url_count": 0}})
    assert next_state(State.UPLOADING, Event(stage, state, pct), snapshot) == State.REVIEW
