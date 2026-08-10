from __future__ import annotations

import json

import numpy as np
import pytest
import requests

from hanging_mvp import exit_review
from hanging_mvp.client_result import build_client_result
from worker import result_builder


class _FakeResponse:
    def __init__(self, content, *, request_id="req-v28"):
        self._content = content
        self.headers = {"x-request-id": request_id}

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "completion-v28",
            "choices": [{"message": {"content": self._content}}],
        }


def _required_candidate():
    return {
        "candidate_id": "cand_000",
        "wall_id": "wall_front_01",
        "score": 0.9,
        "requires_hardpaste_vlm_review": True,
        "hardpaste_vlm_review_fail_closed": True,
        "hardpaste_vlm_review_reasons": ["wall_component_created_by_window_or_opening_mask_subtraction"],
        "hard_constraints": {"passed": True, "hard_gate_reasons": []},
        "projection": {"shape_gate": {"status": "pass", "hard_reasons": []}},
    }


def _configure_generic(monkeypatch):
    for name in ("SPATIAL_VLM_API_URL", "SPATIAL_VLM_API_KEY", "SPATIAL_VLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLM_API_URL", "https://vlm.example/v1/chat/completions")
    monkeypatch.setenv("VLM_API_KEY", "test-secret")
    monkeypatch.setenv("VLM_MODEL", "qwen3.6-35B-A3B")
    monkeypatch.setenv("VLM_TIMEOUT_S", "30")


def test_review_helper_is_defined_and_generic_vlm_fallback_is_callable(monkeypatch):
    _configure_generic(monkeypatch)
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return _FakeResponse('{"approved": true, "confidence": 0.91, "reasons": ["solid wall"]}')

    monkeypatch.setattr(exit_review.requests, "post", fake_post)
    image = np.zeros((32, 48, 3), np.uint8)
    result = exit_review.review_hardpaste_exit(image, _required_candidate(), {})

    assert callable(exit_review._call_openai_compatible_review)
    runtime = exit_review.hardpaste_review_runtime_status()
    assert runtime["helper_available"] is True
    assert runtime["configured"] is True
    assert runtime["contract"].endswith("_v28")
    assert result["status"] == "approved_by_vlm"
    assert result["approved_for_flux_render"] is True
    assert result["provider"] == "generic_vlm"
    assert result["model"] == "qwen3.6-35B-A3B"
    assert result["provider_request_id"] == "completion-v28"
    assert calls[0][0] == "https://vlm.example/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer test-secret"
    assert calls[0][3] == 30.0


def test_review_response_string_false_is_never_truthy_approved(monkeypatch):
    _configure_generic(monkeypatch)
    monkeypatch.setattr(
        exit_review.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            [{"type": "text", "text": '```json\n{"approved":"false","confidence":0.99,"reasons":"curtain"}\n```'}]
        ),
    )
    image = np.zeros((20, 20, 3), np.uint8)
    result = exit_review.review_hardpaste_exit(image, _required_candidate(), {})

    assert result["status"] == "rejected_by_vlm"
    assert result["approved_for_flux_render"] is False
    assert result["failure_category"] == "service"
    assert result["reasons"] == ["curtain"]


def test_review_request_error_is_program_failure_not_service_rejection(monkeypatch):
    _configure_generic(monkeypatch)

    def fail(*args, **kwargs):
        raise requests.ConnectionError("temporary connection failure")

    monkeypatch.setattr(exit_review.requests, "post", fail)
    result = exit_review.review_hardpaste_exit(
        np.zeros((20, 20, 3), np.uint8), _required_candidate(), {}
    )

    assert result["status"] == "vlm_review_error_blocked"
    assert result["error_type"] == "vlm_network_or_http_error"
    assert result["failure_category"] == "program"

    summary = exit_review.summarize_hardpaste_review_failures([
        {
            "asset_id": "wall_front_01__cand_000",
            "wall_id": "wall_front_01",
            "candidate_id": "cand_000",
            "approved_for_flux_render": False,
            "review": result,
        }
    ])
    assert summary["exit_code"] == "FAILED_HARDPASTE_REVIEW_INTERNAL"
    assert summary["category"] == "program"
    assert "temporary connection failure" not in json.dumps(summary, ensure_ascii=False)


def test_missing_review_configuration_has_precise_program_exit(monkeypatch):
    for name in (
        "SPATIAL_VLM_API_URL", "SPATIAL_VLM_API_KEY", "SPATIAL_VLM_MODEL",
        "VLM_API_URL", "VLM_API_KEY", "VLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    result = exit_review.review_hardpaste_exit(
        np.zeros((20, 20, 3), np.uint8), _required_candidate(), {}
    )
    summary = exit_review.summarize_hardpaste_review_failures([
        {"approved_for_flux_render": False, "review": result}
    ])

    assert result["status"] == "pending_vlm_review_not_configured"
    assert summary["exit_code"] == "FAILED_HARDPASTE_REVIEW_UNAVAILABLE"
    assert summary["source"] == "hardpaste_review_configuration"


def test_genuine_vlm_rejection_remains_service_decision():
    summary = exit_review.summarize_hardpaste_review_failures([
        {
            "approved_for_flux_render": False,
            "review": {
                "status": "rejected_by_vlm",
                "failure_category": "service",
            },
        }
    ])
    assert summary is None


def test_client_and_worker_preserve_public_safe_review_program_failure():
    failure = {
        "category": "program",
        "source": "hardpaste_review_runtime",
        "exit_code": "FAILED_HARDPASTE_REVIEW_INTERNAL",
        "public_reason": "挂画位置已经完成计算，但视觉复核服务暂时未能完成。",
        "public_suggestion": "问题已经记录，我们将在今日内进行人工处理；完成后会通过消息通知，您也可以在历史记录中查看结果。",
    }
    plan = {
        "status": "failed",
        "exit_code": failure["exit_code"],
        "exit_review_failure": failure,
        "final_hardpaste_assets": [],
        "image_space": {"original_size": [100, 100]},
    }
    client = build_client_result(plan, [], [])
    assert client["exit_review_failure"] == failure

    ctx = result_builder.build_failure_context(
        client["exit_code"],
        "failed",
        stage="render",
        failure_detail=client["exit_review_failure"],
    )
    assert ctx["category"] == "program"
    assert ctx["source"] == "hardpaste_review_runtime"
    assert "今日内" in ctx["public_suggestion"]
    assert "历史记录" in ctx["public_suggestion"]
