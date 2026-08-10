from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Any, Mapping, Sequence

import requests
from PIL import Image

from .rules_access import rule_value

logger = logging.getLogger(__name__)


_REVIEW_PROGRAM_ERROR_STATUSES = {
    "pending_vlm_review_not_configured",
    "blocked_unreviewed",
    "vlm_review_error_blocked",
    "hardpaste_export_error",
}


def _image_array_to_data_url(image_rgb) -> str:
    img = Image.fromarray(image_rgb.astype("uint8")).convert("RGB")
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("review VLM did not return a JSON object")
    return data


def _review_vlm_config() -> dict[str, Any]:
    """Resolve the hard-paste review VLM from one deterministic env contract.

    The spatial expert is preferred, while the generic OpenAI-compatible VLM is
    a supported fallback.  Keeping resolution in one place avoids the merge bug
    that previously left the request helper body unreachable after ``return``.
    """
    spatial_url = os.getenv("SPATIAL_VLM_API_URL", "").strip()
    spatial_key = os.getenv("SPATIAL_VLM_API_KEY", "").strip()
    spatial_model = os.getenv("SPATIAL_VLM_MODEL", "").strip()
    generic_url = os.getenv("VLM_API_URL", "").strip()
    generic_key = os.getenv("VLM_API_KEY", "").strip()
    generic_model = os.getenv("VLM_MODEL", "").strip()

    spatial_complete = bool(spatial_url and spatial_key and spatial_model)
    generic_complete = bool(generic_url and generic_key and generic_model)
    spatial_present = bool(spatial_url or spatial_key or spatial_model)
    if spatial_complete:
        use_spatial = True
    elif generic_complete:
        use_spatial = False
    else:
        # Preserve the incomplete group in diagnostics instead of silently
        # combining credentials/model fields from two different providers.
        use_spatial = spatial_present
    api_url = spatial_url if use_spatial else generic_url
    api_key = spatial_key if use_spatial else generic_key
    model = spatial_model if use_spatial else generic_model
    timeout_raw = (
        os.getenv("SPATIAL_VLM_TIMEOUT_S", "").strip()
        if use_spatial
        else os.getenv("VLM_TIMEOUT_S", "").strip()
    ) or os.getenv("VLM_TIMEOUT_S", "120").strip() or "120"
    try:
        timeout_s = max(1.0, float(timeout_raw))
    except (TypeError, ValueError):
        raise ValueError(f"invalid hard-paste review VLM timeout: {timeout_raw!r}")

    source = "spatial_vlm" if use_spatial else "generic_vlm"
    return {
        "api_url": api_url,
        "api_key": api_key,
        "model": model,
        "timeout_s": timeout_s,
        "source": source,
        "configured": bool(api_url and api_key and model),
    }


def _spatial_vlm_configured() -> bool:
    # Historical function name retained for compatibility with existing tests
    # and callers.  It now means that either supported review VLM is complete.
    return bool(_review_vlm_config()["configured"])


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    raise ValueError("review VLM response message content is not text")


def _coerce_approved(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) == 1.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "approved", "approve", "1"}:
            return True
        if normalized in {"false", "no", "rejected", "reject", "0", ""}:
            return False
    raise ValueError("review VLM field 'approved' must be a boolean")


def _coerce_confidence(value: Any) -> float:
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("review VLM confidence must be between 0 and 1")
    return confidence


def _coerce_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("review VLM field 'reasons' must be a string or list")


def _call_openai_compatible_review(image_rgb, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Call the configured OpenAI-compatible VLM for hard-paste exit QA."""
    cfg = _review_vlm_config()
    if not cfg["configured"]:
        raise RuntimeError(
            "hard-paste review VLM is not configured: set either "
            "SPATIAL_VLM_API_URL/API_KEY/MODEL or VLM_API_URL/API_KEY/MODEL"
        )

    prompt = {
        "task": "Review whether this hard-pasted artwork placement is safe to send to final photorealistic rendering.",
        "candidate": {
            "candidate_id": candidate.get("candidate_id"),
            "wall_id": candidate.get("wall_id"),
            "risk_level": candidate.get("risk_level"),
            "score": candidate.get("score"),
            "soft_risk_reasons": candidate.get("soft_risk_reasons", []),
            "hardpaste_vlm_review_reasons": candidate.get("hardpaste_vlm_review_reasons", []),
            "wall_mask_lineage_gate": candidate.get("wall_mask_lineage_gate"),
        },
        "approve_when": [
            "artwork is visibly on a solid wall, not on a door, window, mirror, floor, ceiling, person, opening or major obstruction",
            "artwork is not severely clipped by image boundary",
            "perspective and scale are plausible enough for a final image edit model to refine",
        ],
        "reject_when": [
            "artwork is placed on a structural object, opening, person or impossible geometry",
            "artwork is placed on a curtain, sheer curtain, drape, blind, window covering or other non-rigid surface even if it looks planar",
            "a landscape artwork is visibly warped into portrait orientation, a portrait artwork into landscape orientation, or the perspective foreshortening is implausibly severe",
            "major part of the artwork is off-wall or occluded",
            "hard paste creates an obviously invalid target for final rendering",
        ],
        "return_json_only": {
            "approved": True,
            "confidence": 0.0,
            "reasons": ["..."],
        },
    }
    payload = {
        "model": cfg["model"],
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict visual QA reviewer for wall-art placement. "
                    "Return one JSON object only. approved must be a JSON boolean and "
                    "confidence must be a number from 0 to 1."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _image_array_to_data_url(image_rgb)}},
                ],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        cfg["api_url"],
        headers=headers,
        json=payload,
        timeout=cfg["timeout_s"],
    )
    resp.raise_for_status()
    body = resp.json()
    text = _message_content_to_text(body["choices"][0]["message"]["content"])
    data = _extract_json(text)
    return {
        "approved": _coerce_approved(data.get("approved")),
        "confidence": _coerce_confidence(data.get("confidence")),
        "reasons": _coerce_reasons(data.get("reasons")),
        "provider": cfg["source"],
        "model": cfg["model"],
        "provider_request_id": body.get("id") or resp.headers.get("x-request-id") or resp.headers.get("request-id"),
    }


def hardpaste_review_runtime_status() -> dict[str, Any]:
    """Return a secret-free health payload for deployment verification."""
    try:
        cfg = _review_vlm_config()
        return {
            "contract": "openai_compatible_hardpaste_exit_review_v28",
            "helper_available": callable(globals().get("_call_openai_compatible_review")),
            "configured": bool(cfg.get("configured")),
            "provider_source": str(cfg.get("source") or ""),
            "model": str(cfg.get("model") or ""),
            "configuration_error": None,
        }
    except Exception as exc:  # configuration status only; no credentials returned
        return {
            "contract": "openai_compatible_hardpaste_exit_review_v28",
            "helper_available": callable(globals().get("_call_openai_compatible_review")),
            "configured": False,
            "provider_source": "",
            "model": "",
            "configuration_error": type(exc).__name__,
        }


def summarize_hardpaste_review_failures(review_records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return a public-safe program failure when review infrastructure failed.

    A genuine VLM rejection remains a service decision.  Missing configuration,
    request/response errors, internal exceptions, or hard-paste export failures
    are program failures and must never be presented as "no safe wall".
    """
    records = [record for record in review_records if isinstance(record, Mapping)]
    if not records or any(bool(record.get("approved_for_flux_render")) for record in records):
        return None

    failures: list[dict[str, str]] = []
    for record in records:
        review = record.get("review") if isinstance(record.get("review"), Mapping) else {}
        status = str(review.get("status") or "").strip()
        if status not in _REVIEW_PROGRAM_ERROR_STATUSES:
            continue
        failures.append({
            "asset_id": str(record.get("asset_id") or ""),
            "wall_id": str(record.get("wall_id") or ""),
            "candidate_id": str(record.get("candidate_id") or ""),
            "status": status,
            "error_type": str(review.get("error_type") or ""),
        })

    if not failures:
        return None

    statuses = {item["status"] for item in failures}
    error_types = {item["error_type"] for item in failures if item["error_type"]}
    if "hardpaste_export_error" in statuses:
        source = "hardpaste_export"
        exit_code = "FAILED_HARDPASTE_EXPORT"
    elif statuses <= {"pending_vlm_review_not_configured", "blocked_unreviewed"}:
        source = "hardpaste_review_configuration"
        exit_code = "FAILED_HARDPASTE_REVIEW_UNAVAILABLE"
    else:
        source = "hardpaste_review_runtime"
        exit_code = "FAILED_HARDPASTE_REVIEW_INTERNAL"

    return {
        "category": "program",
        "source": source,
        "exit_code": exit_code,
        "failed_review_count": len(failures),
        "review_statuses": sorted(statuses),
        "error_types": sorted(error_types),
        # No provider body, stack trace, URL, API key, or raw exception is copied.
        "failed_reviews": failures,
        "public_reason": "挂画位置已经完成计算，但视觉复核服务暂时未能完成。",
        "public_suggestion": "问题已经记录，我们将在今日内进行人工处理；完成后会通过消息通知，您也可以在历史记录中查看结果。",
    }


def review_hardpaste_exit(image_rgb, candidate: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Exit gate for hard-pasted assets before they enter final rendering.

    Low-risk candidates pass by metadata. High-risk candidates can be reviewed by
    an OpenAI-compatible VLM when configured. Unreviewed high-risk candidates are
    blocked by default so the render batch only sees approved hard-pastes.
    """
    hard_constraints = candidate.get("hard_constraints") if isinstance(candidate.get("hard_constraints"), Mapping) else {}
    projection_guard = hard_constraints.get("projection_degenerate_guard") if isinstance(hard_constraints.get("projection_degenerate_guard"), Mapping) else {}
    projection_shape = projection_guard.get("projection_shape_gate") if isinstance(projection_guard.get("projection_shape_gate"), Mapping) else {}
    if not projection_shape:
        projection = candidate.get("projection") if isinstance(candidate.get("projection"), Mapping) else {}
        projection_shape = projection.get("shape_gate") if isinstance(projection.get("shape_gate"), Mapping) else {}
    numeric_hard_reasons = list(dict.fromkeys(
        [str(x) for x in candidate.get("reject_reasons", []) or [] if str(x)]
        + [str(x) for x in hard_constraints.get("hard_gate_reasons", []) or [] if str(x)]
        + [str(x) for x in projection_shape.get("hard_reasons", []) or [] if str(x)]
    ))
    numeric_hard_reject = bool(
        hard_constraints.get("passed") is False
        or projection_shape.get("status") == "hard_reject"
        or numeric_hard_reasons
    )
    # Deterministic metric hard gates are one-way: no visual model,
    # configuration mode or confidence score may turn them back into an
    # approved render input.
    if numeric_hard_reject:
        return {
            "enabled": True,
            "required": False,
            "fail_closed_required": True,
            "mode": "numeric_geometry_gate",
            "status": "blocked_by_numeric_geometry_hard_gate",
            "approved": False,
            "approved_for_flux_render": False,
            "confidence": None,
            "reasons": numeric_hard_reasons or ["hard_constraints_failed"],
            "visual_review_can_override": False,
            "failure_category": "service",
        }

    fail_closed_required = bool(candidate.get("hardpaste_vlm_review_fail_closed"))
    requires_review = bool(candidate.get("requires_hardpaste_vlm_review") or fail_closed_required)
    review_all = bool(rule_value(rules, ["exit_review", "review_all_hardpaste"], False))
    min_conf = float(rule_value(rules, ["exit_review", "min_vlm_confidence"], 0.55))
    block_unreviewed = bool(rule_value(rules, ["exit_review", "block_unreviewed_required_vlm"], True)) or fail_closed_required
    mode = str(rule_value(rules, ["exit_review", "hardpaste_vlm_mode"], os.getenv("HARDPASTE_EXIT_REVIEW_MODE", "auto"))).strip().lower()

    if not requires_review and not review_all:
        return {
            "enabled": True,
            "required": False,
            "fail_closed_required": False,
            "mode": mode,
            "status": "approved_by_geometry_gate",
            "approved": True,
            "approved_for_flux_render": True,
            "confidence": None,
            "reasons": [],
        }

    if mode in {"off", "disabled", "metadata_only"}:
        approved = not (requires_review and block_unreviewed)
        return {
            "enabled": True,
            "required": requires_review,
            "fail_closed_required": fail_closed_required,
            "mode": mode,
            "status": "blocked_unreviewed" if not approved else "approved_without_vlm_by_rule",
            "approved": approved,
            "approved_for_flux_render": approved,
            "confidence": None,
            "reasons": ["vlm_review_disabled"],
            "error_type": "vlm_review_disabled_by_configuration" if not approved else None,
            "failure_category": "program" if not approved else None,
        }

    try:
        config = _review_vlm_config()
    except (TypeError, ValueError) as exc:
        logger.warning("hardpaste exit VLM configuration is invalid", exc_info=True)
        return {
            "enabled": True,
            "required": requires_review,
            "fail_closed_required": fail_closed_required,
            "mode": mode,
            "status": "vlm_review_error_blocked",
            "approved": False,
            "approved_for_flux_render": False,
            "confidence": None,
            "error_type": "vlm_response_or_configuration_error",
            "failure_category": "program",
            "reasons": [str(exc)],
        }

    if not config["configured"]:
        approved = not (requires_review and block_unreviewed)
        return {
            "enabled": True,
            "required": requires_review,
            "fail_closed_required": fail_closed_required,
            "mode": mode,
            "status": "pending_vlm_review_not_configured" if not approved else "approved_without_vlm_no_config",
            "approved": approved,
            "approved_for_flux_render": approved,
            "confidence": None,
            "reasons": [
                "set either SPATIAL_VLM_API_URL/API_KEY/MODEL or VLM_API_URL/API_KEY/MODEL"
            ],
            "error_type": "vlm_not_configured" if not approved else None,
            "failure_category": "program" if not approved else None,
        }

    try:
        data = _call_openai_compatible_review(image_rgb, candidate)
        confidence = float(data["confidence"])
        approved = bool(data["approved"]) and confidence >= min_conf
        return {
            "enabled": True,
            "required": requires_review,
            "fail_closed_required": fail_closed_required,
            "mode": "openai_compatible",
            "status": "approved_by_vlm" if approved else "rejected_by_vlm",
            "approved": approved,
            "approved_for_flux_render": approved,
            "confidence": confidence,
            "min_confidence": min_conf,
            "reasons": data.get("reasons", []),
            "provider": data.get("provider"),
            "model": data.get("model"),
            "provider_request_id": data.get("provider_request_id"),
            "failure_category": None if approved else "service",
        }
    except requests.exceptions.RequestException as exc:
        logger.warning("hardpaste exit VLM request failed", exc_info=True)
        return {
            "enabled": True,
            "required": requires_review,
            "fail_closed_required": fail_closed_required,
            "mode": "openai_compatible",
            "status": "vlm_review_error_blocked",
            "approved": False,
            "approved_for_flux_render": False,
            "confidence": None,
            "error_type": "vlm_network_or_http_error",
            "failure_category": "program",
            "reasons": [str(exc)],
        }
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("hardpaste exit VLM response parsing failed", exc_info=True)
        return {
            "enabled": True,
            "required": requires_review,
            "fail_closed_required": fail_closed_required,
            "mode": "openai_compatible",
            "status": "vlm_review_error_blocked",
            "approved": False,
            "approved_for_flux_render": False,
            "confidence": None,
            "error_type": "vlm_response_or_configuration_error",
            "failure_category": "program",
            "reasons": [str(exc)],
        }
    except Exception as exc:
        logger.exception("hardpaste exit VLM failed unexpectedly")
        return {
            "enabled": True,
            "required": requires_review,
            "fail_closed_required": fail_closed_required,
            "mode": "openai_compatible",
            "status": "vlm_review_error_blocked",
            "approved": False,
            "approved_for_flux_render": False,
            "confidence": None,
            "error_type": "vlm_unexpected_error",
            "failure_category": "program",
            "reasons": [str(exc)],
        }
