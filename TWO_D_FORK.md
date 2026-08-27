# 2-D sampling fork

Starting point: consolidated `main` at `d695d8d`.

Gate 6 found that in one dimension, a continuous boundary-to-boundary shuttle can nearly match IID-random coverage while moving roughly thirty times less internal distance. It also found that delayed event addresses must live in the current trajectory coordinate frame.

This branch attacks the likely loophole:

> **1-D intervals make the shuttle geometrically obvious. What replaces it in two dimensions?**

## Equal-budget competitors

Every continuous path gets:

- 100 probes per 100-ms cycle;
- persistent trajectory state across cycle boundaries;
- the same noisy 2-D directional/control cue;
- the same uncertainty-radius rule;
- exactly the same nominal internal path budget: `4 * radius` per cycle;
- no slow learning.

Competitors:

- persistent boustrophedon/raster;
- persistent Hilbert order-3 curve;
- square spiral;
- Lissajous path;
- golden-angle radial spokes;
- Halton points connected by a greedy short tour;
- matched-step smooth random walk;
- point estimate;
- unmatched IID-random upper bound.

Worlds:

1. reliable moving cue;
2. changing cue reliability;
3. systematic 2-D cue bias;
4. cue loss and return.

The first pass asks only geometry/coverage. A later pass will address delayed events on the winning 2-D trajectory if any structured path survives.

## Pass condition

A candidate should not be called useful because it wins one toy world. It should:

1. beat the matched-travel smooth random walk in multiple regimes;
2. approach IID-random hit coverage without IID-random travel;
3. remain useful under systematic cue bias or cue loss;
4. preserve continuous state across cycles.

If no deterministic trajectory does that, the strong 1-D result does not generalize.
