from worker import result_builder


def test_missing_artwork_dimensions_is_service_failure():
    ctx = result_builder.build_failure_context(
        "FAILED_MISSING_ARTWORK_DIMENSIONS", "failed", stage="hanging"
    )
    assert ctx["category"] == "service"
    assert ctx["source"] == "artwork_dimensions"


def test_no_safe_wall_preserves_public_safe_reason():
    ctx = result_builder.build_failure_context(
        "PARTIAL_ALL_REJECTED_WITH_REASONS",
        "no_safe_wall",
        not_recommended=[{"reasons": ["窗帘安全间距不足"]}],
        stage="hanging",
    )
    assert ctx["category"] == "service"
    assert ctx["source"] == "safe_area"
    assert ctx["public_reason"] == "窗帘安全间距不足"


def test_render_review_is_service_quality_failure():
    ctx = result_builder.build_failure_context(
        "PARTIAL_NO_FLUX_APPROVED_CANDIDATE", "render_review", stage="render"
    )
    assert ctx["category"] == "service"
    assert ctx["source"] == "render_quality"


def test_worker_exception_is_program_failure():
    ctx = result_builder.build_failure_context(
        "FAILED_WORKER_EXCEPTION", "failed", stage="worker"
    )
    assert ctx["category"] == "program"
    assert ctx["source"] == "worker"
    assert "Traceback" not in str(ctx)


def test_failed_payload_exposes_failure_context():
    payload = result_builder.build_failed_payload(
        {"job_id": "job_1", "order_id": "order_1", "render_provider": "apiyi"},
        {"exit_code": "FAILED_MISSING_ARTWORK_DIMENSIONS", "not_recommended": []},
    )
    assert payload["failure_context"]["source"] == "artwork_dimensions"
