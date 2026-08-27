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


## Fork 1 result — no universal 2-D path

10-seed means:

```text
RELIABLE
IID random, unmatched       100.0%   travel 0.581
golden radial                 90.2%   travel 0.0222
point estimate                79.8%
Halton short tour             60.5%
Hilbert                       54.5%
smooth random walk            26.8%

MIXED RELIABILITY
IID random                    91.2%
golden radial                62.3%
Halton short tour            46.7%
point estimate               41.7%
smooth random walk           17.8%

SYSTEMATIC CUE BIAS
IID random                    97.3%
Lissajous                     54.3%
square spiral                 54.0%
boustrophedon                 42.5%
golden radial                 36.0%
point estimate                 3.7%

LOSS / RETURN
IID random                     0 ms
golden radial                 36.7 ms
Halton short tour             56.7 ms
boustrophedon                120 ms
point estimate               126.7 ms
smooth random walk           340 ms
```

The 1-D result generalizes only partially:

> **structured low-travel paths still beat generic smooth motion in 2-D, but the best geometry depends on the uncertainty/failure mode.**

Radial coverage is strong when the cue is approximately centered but noisy. Spiral/Lissajous coverage is much stronger when the cue is systematically biased.

Receipt: `results/fork_2d_sampling_geometry.json`.

## Fork 2 — let failure switch geometry

The next question is architectural rather than geometric:

> **Can failure of the current sampler become a fast state that changes how the machine samples, without changing slow weights?**

Each cycle returns one binary hit/miss consequence. An EMA-like fast miss state may:

- keep golden radial sampling while recent cycles succeed;
- switch to spiral or Lissajous after repeated misses;
- expand the coverage radius without receiving extra movement budget;
- return to radial sampling after success.

The path remains continuous when geometry changes. The nominal movement budget still depends only on cue confidence, so a failure-driven expansion cannot buy extra travel.

Implementation: `experiments/fork_2d_adaptive_geometry.py`.


## Fork 2 result — binary miss switching is too crude

The fast miss-state controller really did switch geometry, but it usually hurt.

```text
RELIABLE
fixed radial                  90.0%
adaptive radial -> spiral     72.1%

MIXED
fixed radial                  61.0%
adaptive radial -> spiral     43.5%

SYSTEMATIC BIAS
fixed spiral                  54.0%
adaptive radial -> spiral     40.7%
fixed radial                  36.8%

LOSS / RETURN
fixed radial                  41.7 ms
adaptive radial -> spiral    194.4 ms
```

The failure is informative:

> **"I missed" says that the current policy is inadequate, but not how the internal reference is wrong.**

Switching geometry also disrupts useful path persistence.

Receipt: `results/fork_2d_adaptive_geometry.json`.

## Fork 3 — let relevant samples calibrate the fast reference

Instead of using a miss to choose a new path, keep the strong radial sampler and use *where relevant samples occurred* to update a temporary 2-D calibration offset.

Two local rules are attacked:

- hit-only recentering: successful sample offsets pull the fast center;
- contrast recentering: within one sweep, samples better than the sweep mean pull the fast center while worse samples push against it.

The system is not given target direction or a derivative. The fast offset decays and is never consolidated.

This is the 2-D version of the earlier fast-state steering idea, but with a narrower job: **calibrate an already useful directional cue rather than discover the target from scratch**.

Implementation: `experiments/fork_2d_fast_recenter.py`.
