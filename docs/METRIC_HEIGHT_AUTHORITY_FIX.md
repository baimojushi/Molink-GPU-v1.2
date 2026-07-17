# Metric-height authority and high-placement fix

## Regression

The failing candidate declared `table_center_135` but its final center was
2.121 m above the floor. A low-coverage fallback found a high empty rectangle
and wrote `center_t_m_override=0.744`, silently replacing the 1.35 m metric
rule by 0.771 m. The client simultaneously reported `center_t_label=偏高` and
`高度符合规则`.

Job `job_461299b7` exposed the opposite failure mode in V2: the 1.52 m metric
target was preserved, but a high max-empty rectangle rejected the otherwise
perfect candidate because the target was outside that rectangle. V3 removes
that false hard gate. The rectangle remains horizontal seed evidence only.

## Contract

- Empty-space geometry may select a horizontal axis only.
- A finalized metric height is authoritative and cannot be rewritten by an
  automatic empty-rectangle center.
- A max-empty rectangle is a horizontal seed only. Its vertical interval is
  diagnostic and cannot move or reject a trusted metric target.
- Candidate acceptance is decided from the artwork footprint projected at the
  metric target: wall/edge/corner support, structural clearance, blockers and
  occlusion remain authoritative safety gates.
- Support objects must have enough points close to the current wall plane;
  foreground furniture on another depth plane cannot choose the height rule.
- When no wall-associated furniture survives and the floor anchor is trusted,
  use the standard floor-center rule.

The plan and client result export `pipeline_build`, while every candidate
exports `height_rule_consistency` and `semantic_clear_rect_vertical_fit` so a
stale deployment or future contract violation is directly observable.
