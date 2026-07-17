# Furniture support height fix (V5)

## Regression

`job_5b71f65a` selected `floor_center_152` although the chosen front-wall axis
had furniture/counter geometry below the artwork. The serialized semantic JSON
also omitted masks, but the actual runtime result proves that `table_07` had
11,714 sampled points and was correctly rejected as distant foreground
(`wall_distance_p20_m=6.55`). The real wall-adjacent counter was not labeled.

## Contract

- A semantic support with a SAM mask keeps the strict current-wall association
  gate.
- When only a processed 2D bbox is available, sample its conservative inner
  region and still require metric depth association with the current wall.
- When no semantic furniture is associated, inspect unlabeled geometry near
  the candidate axis, excluding both the wall plane and distant foreground.
- A dense, broad height band in the plausible furniture range may provide a
  generic wall-adjacent support top.
- Furniture context never lowers the trusted floor default. It raises the
  center using label-specific bottom clearance: table 0.20m and
  counter/cabinet/sideboard/generic geometry support 0.25m.

V4 successfully recognized the `job_5b71f65a` bar as `geometry_support` at
1.168m above floor, but its global 0.15m clearance resolved to 1.518m and was
therefore clamped to the 1.52m floor default. V5 uses the cabinet/counter
clearance of 0.25m, resolving that same evidence to about 1.618m.

This preserves the V2 protection against foreground tables while recovering
missed counters/cabinets and preventing furniture-adjacent artwork from being
placed too low.
