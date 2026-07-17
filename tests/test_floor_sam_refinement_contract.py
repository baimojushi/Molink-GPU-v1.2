from __future__ import annotations

from semantic_mvp.sam_refiner import default_refine_label_set


def test_floor_remains_in_sam_refine_set_when_override_omits_it(monkeypatch):
    monkeypatch.setenv("SAM_REFINE_LABELS", "door,window,existing_art")
    monkeypatch.delenv("SAM_FORCE_REFINE_LABELS", raising=False)
    assert "floor" in default_refine_label_set()


def test_operator_can_explicitly_disable_forced_floor_refinement(monkeypatch):
    monkeypatch.setenv("SAM_REFINE_LABELS", "door,window")
    monkeypatch.setenv("SAM_FORCE_REFINE_LABELS", "")
    assert "floor" not in default_refine_label_set()
