from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Any, Mapping

import requests
from PIL import Image

from .rules_access import rule_value

logger = logging.getLogger(__name__)


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


def _vlm_configured() -> bool:
    return bool(os.getenv("VLM_API_URL", "").strip() and os.getenv("VLM_API_KEY", "").strip() and os.getenv("VLM_MODEL", "").strip())


def _call_openai_compatible_review(image_rgb, candidate: Mapping[str, Any]) -> dict[str, Any]:
    api_url = os.getenv("VLM_API_URL", "").strip()
    api_key = os.getenv("VLM_API_KEY", "").strip()
    model = os.getenv("VLM_MODEL", "").strip()
    prompt = {
        "task": "Review whether this hard-pasted artwork placement is safe to send to final photorealistic Flux rendering.",
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
        "return_json_only": {"approved": True, "confidence": 0.0, "reasons": ["..."]},
    }
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a strict visual QA reviewer for wall-art placement. Return JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _image_array_to_data_url(image_rgb)}},
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=float(os.getenv("VLM_TIMEOUT_S", "120")))
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _extract_json(text)


def review_hardpaste_exit(image_rgb, candidate: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Exit gate for hard-pasted assets before they enter Flux rendering.

    Low-risk candidates pass by metadata. High-risk candidates can be reviewed by
    an OpenAI-compatible VLM when configured. Unreviewed high-risk candidates are
    blocked by default so the Flux batch script only sees approved hard-pastes.
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
    # ROOT CAUSE / DO NOT REGRESS (2026-07-21): the failed hard-paste had
    # 1.85x aspect distortion, but the former threshold classified it as a
    # visual-review item and a free-text VLM approved it.  Deterministic metric
    # hard gates are one-way: no visual model, configuration mode or confidence
    # score may turn them back into an approved Flux input.
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
        }

    if not _vlm_configured():
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
            "reasons": ["VLM_API_URL, VLM_API_KEY and VLM_MODEL are not all configured"],
        }

    try:
        data = _call_openai_compatible_review(image_rgb, candidate)
        confidence = float(data.get("confidence", 0.0) or 0.0)
        approved = bool(data.get("approved")) and confidence >= min_conf
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
            "reasons": [str(exc)],
        }
