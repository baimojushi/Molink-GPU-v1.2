from __future__ import annotations

import ast
from pathlib import Path


def test_service_imports_refactored_helpers_and_keeps_endpoint():
    root = Path(__file__).resolve().parents[1]
    service_py = root / "hanging_mvp" / "service.py"
    tree = ast.parse(service_py.read_text(encoding="utf-8"))

    async_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)}
    assert "analyze_hanging_mvp" in async_functions

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert "rules_access" in imports
    assert "structural_clearance" in imports
    assert "support_context" in imports
    assert "response_assets" in imports


def test_candidate_contract_notes_preserve_extension_fields():
    root = Path(__file__).resolve().parents[1]
    notes = (root / "HANGING_MVP_API_CONTRACT.md").read_text(encoding="utf-8")
    for field in [
        "risk_level",
        "soft_risk_reasons",
        "requires_hardpaste_vlm_review",
        "structural_clearance",
        "cleanup_scope",
        "flux_visual_refinement",
        "requires_flux_visual_height_refinement",
    ]:
        assert field in notes


def test_projected_duplicate_suppression_keeps_best_same_wall_candidate():
    from hanging_mvp.service import _suppress_projected_duplicate_candidates

    base_projection = {"projected_quad_px_original": [[100, 10], [200, 10], [200, 90], [100, 90]]}
    candidates = [
        {
            "candidate_id": "best",
            "wall_id": "wall_02",
            "score": 0.90,
            "projection": base_projection,
            "axis": {"axis_source": "solid_component_one_third"},
        },
        {
            "candidate_id": "duplicate",
            "wall_id": "wall_02",
            "score": 0.80,
            "projection": {"projected_quad_px_original": [[125, 10], [225, 10], [225, 90], [125, 90]]},
            "axis": {"axis_source": "semantic_clear_subregion_one_quarter"},
        },
        {
            "candidate_id": "other_wall",
            "wall_id": "wall_03",
            "score": 0.70,
            "projection": base_projection,
            "axis": {"axis_source": "solid_component_center"},
        },
    ]
    kept, dropped, summary = _suppress_projected_duplicate_candidates(candidates, {})
    assert [x["candidate_id"] for x in kept] == ["best", "other_wall"]
    assert [x["candidate_id"] for x in dropped] == ["duplicate"]
    assert summary["dropped_count"] == 1
    assert dropped[0]["selection_status"] == "dropped_by_projected_duplicate_suppression"
