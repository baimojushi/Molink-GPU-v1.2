from __future__ import annotations

from semantic_mvp.sam_refiner import canonical_refine_label, default_refine_label_set


def test_floor_remains_in_sam_refine_set_when_override_omits_it(monkeypatch):
    monkeypatch.setenv("SAM_REFINE_LABELS", "door,window,existing_art")
    monkeypatch.delenv("SAM_FORCE_REFINE_LABELS", raising=False)
    assert "floor" in default_refine_label_set()


def test_operator_can_explicitly_disable_forced_floor_refinement(monkeypatch):
    monkeypatch.setenv("SAM_REFINE_LABELS", "door,window")
    monkeypatch.setenv("SAM_FORCE_REFINE_LABELS", "")
    assert "floor" not in default_refine_label_set()


def test_instance_style_wall_labels_use_wall_sam_policy():
    assert canonical_refine_label("wall_front_01", "wall") == "wall"
    assert canonical_refine_label("wall_right_02") == "wall"
    assert canonical_refine_label("small_shelf", "existing_content") == "small_shelf"
