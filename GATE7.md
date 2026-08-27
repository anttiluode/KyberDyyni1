# Gate 7 — 2-D structured probing and fast-to-slow calibration

Gate 7 attacks the one-dimensional geometry that made Gate 6 look unusually clean.

## Question

> **Does the architecture survive when the uncertainty region is two-dimensional, and what part of the 1-D story remains once ordinary probe-basis attackers are included?**

## 7A — no universal 2-D trajectory

Continuous samplers received equal wall time, probe count and nominal path budget.

10-seed first-pass means:

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
golden radial                 62.3%
Halton short tour             46.7%
point estimate               41.7%
smooth random walk           17.8%

SYSTEMATIC CUE BIAS
IID random                    97.3%
Lissajous                     54.3%
square spiral                 54.0%
golden radial                 36.0%
point estimate                 3.7%
```

The 1-D result therefore generalizes only partially.

> **Structured low-travel sampling still beats generic smooth motion, but the best 2-D geometry depends on the uncertainty/failure mode.**

## 7B — binary miss-driven geometry switching fails

A fast hit/miss EMA was allowed to switch between radial and spiral/Lissajous sampling.

It usually made performance worse.

```text
RELIABLE
fixed radial                  90.0%
adaptive radial -> spiral     72.1%

MIXED
fixed radial                  61.0%
adaptive radial -> spiral     43.5%

LOSS / RETURN
fixed radial                  41.7 ms
adaptive radial -> spiral    194.4 ms
```

A miss says the current sampler failed, but does not provide a direction for correction. Switching also disrupts useful path persistence.

## 7C — sample-relative relevance creates useful fast calibration

Keep the structured radial sampler and update only a temporary 2-D correction vector.

Within each sweep, samples are compared to the sweep mean:

```text
advantage_i = value_i - mean(value)
delta_fast  ~ sum(advantage_i * local_offset_i)
```

No target direction or derivative is provided.

12-seed means:

```text
                         fixed radial    + fast contrast

RELIABLE                    90.0%            92.8%
MIXED                       61.0%            82.5%
SYSTEMATIC BIAS             36.8%            85.1%

LOSS / RETURN
hit-cycle fraction          78.5%            86.3%
reacquisition               41.7 ms          30.6 ms
```

The path-travel budget is unchanged.

The same local rule on a matched-travel smooth random walk is much worse.

Earned:

> **Structured local samples can calibrate an already-useful but imperfect reference through fast elastic state.**

## 7D — delayed consequence crystallizes repeated calibration

Six contexts have stable cue-bias vectors but new target positions every encounter.

The fast sampler solves the bias during the encounter. The successful temporary correction is retained as a local packet and delivered to slow learning only after three later encounters.

12-seed means:

```text
                              FIRST        LATE

bounded delayed calibration
start error                   0.435        0.231
first-cycle hit               51.4%        82.6%
first-cycle best distance     0.223        0.118

fast only / slow frozen
start error                   0.435        0.429
first-cycle hit               51.4%        38.2%

shuffled delayed context
start error                   0.438        0.455
first-cycle hit               50.0%        41.7%
```

The slow learner changes no weights inside the encounter.

This earns:

> **A temporary 2-D calibration discovered by fast structured sampling can become delayed context-specific prior calibration without BPTT.**

The explicit EMA attacker is numerically identical to the bounded learner in this setup, so the current slow rule has no special engineering advantage.

## 7E — ordinary probe-basis attacker nearly matches radial sampling

The radial path was attacked with conventional finite-difference-like probes using the same contrast rule and travel budget.

```text
                         radial     cardinal cross   rotating cross

RELIABLE                 92.8%          90.7%           91.0%
MIXED                    82.5%          81.4%           80.7%
SYSTEMATIC BIAS          85.1%          86.9%           86.0%
LOSS reacquisition       30.6 ms        63.9 ms         63.9 ms
```

SPSA-like one-direction-per-cycle probing is weaker. A perimeter-only ring is very poor. Center-crossing balanced directions matter.

So the surviving digital primitive is not golden-angle sampling:

> **structured local probing around a stable reference**

is enough.

## Consolidated Gate-7 architecture

```text
             STABLE / SLOW REFERENCE
                       |
             approximate fast cue
                       |
                       v
             STRUCTURED LOCAL PROBES
              +/- useful directions
                       |
              relative relevance
                       |
                       v
             FAST CORRECTION STATE
                       |
             immediate better sampling
                       |
                  ... later ...
                       |
             context-local consequence
                       |
                       v
              SLOW CALIBRATION
```

This is increasingly conventional engineering: zeroth-order probing + elastic state + delayed associative memory.

That simplification is an earned result.

## Canonical files

- `experiments/fork_2d_sampling_geometry.py`
- `results/fork_2d_sampling_geometry.json`
- `experiments/fork_2d_adaptive_geometry.py`
- `results/fork_2d_adaptive_geometry.json`
- `experiments/fork_2d_fast_recenter.py`
- `results/fork_2d_fast_recenter.json`
- `experiments/fork_2d_fast_slow_calibration.py`
- `results/fork_2d_fast_slow_calibration.json`
- `experiments/fork_2d_probe_basis_attack.py`
- `results/fork_2d_probe_basis_attack.json`
- [TWO_D_CONCLUSION.md](TWO_D_CONCLUSION.md)

## Still not earned

- a general high-dimensional AI primitive;
- sublinear probe cost in latent dimension;
- superiority over standard finite differences or SPSA;
- a special role for biological theta in the digital implementation;
- superiority of the bounded slow learner over an explicit context table.
