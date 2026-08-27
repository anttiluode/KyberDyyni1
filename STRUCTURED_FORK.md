# Structured sampling fork

Starting point: the frozen `control-law-fork` conclusion at `7f086f0`.

This fork attacks the operation that survived the previous branch:

> **structured low-travel sampling around an uncertain control vector**

rather than assuming left/right theta alternation is special.

## Fork 1 — equal-path closed-cycle schedules

All continuous schedules received:

- equal wall time;
- 100 probes per 100-ms cycle;
- the same noisy directional cue;
- no slow learning;
- exactly the same per-cycle total-variation/path budget.

IID random remained an intentionally unmatched high-travel upper bound.

Selected hit-cycle means:

```text
RELIABLE
IID random, unmatched          100.0%   travel 0.327
bilateral triangle              98.3%   travel 0.0113
bilateral sine                  97.5%   travel 0.0113
sorted VdC coverage             97.2%   travel 0.0113
one-side alternating            92.5%   travel 0.0113

MIXED RELIABILITY
IID random                      89.2%
bilateral sine                  77.8%
sorted VdC                      76.7%
bilateral triangle              76.5%
one-side alternating            71.5%

SYSTEMATIC CUE BIAS
IID random                      99.5%
sorted VdC                      74.3%
bilateral triangle              72.7%
one-side alternating            66.3%
axis only                       33.8%
```

Conclusion:

> **Alternating one side per cycle is not special when every cycle is forced to return to center.**

A bilateral sweep and sorted low-discrepancy coverage use the same path budget better.

Receipt: `results/fork_structured_sequences.json`.

## Fork 2 — let the sampler stay continuous across cycle boundaries

The first comparison had a hidden artificial cost: every schedule was forced to return to center at the cycle boundary.

Fork 2 lets the sampling offset persist.

The key attacker is a monotonic shuttle:

```text
cycle n:      left ----------------------> right
cycle n+1:   right ----------------------> left
```

No return-to-center tax.

Selected results:

```text
RELIABLE
IID random                      100.0%   travel 0.327
cross-cycle shuttle              99.17%  travel 0.0111
closed bilateral                 98.33%
closed one-side                  92.92%

MIXED RELIABILITY
IID random                       90.56%
cross-cycle shuttle              89.31%
closed bilateral                 78.33%
closed one-side                  72.36%

SYSTEMATIC CUE BIAS
IID random                       99.58%
cross-cycle sine                 95.83%
cross-cycle shuttle              95.69%
closed bilateral                 73.89%
axis only                        36.11%

LOSS / RETURN
IID random                        0 ms reacquisition
cross-cycle shuttle               5.6 ms
closed one-side                  19.4 ms
closed bilateral                 27.8 ms
axis only                        80.6 ms
smooth random walk              227.8 ms
```

This is substantially stronger than the previous result.

> **Persisting the sweep endpoint across cycles makes alternating boundary-to-boundary traversal an extremely path-efficient 1-D coverage law.**

It gets close to IID random target coverage while moving roughly thirty times less distance.

Receipt: `results/fork_cross_cycle_sequences.json`.

## Fork 3 — can phase still address events when the sweep reverses?

The cross-cycle result creates a new problem for Gate 2.

Raw theta phase is not a stable spatial coordinate under a reversing sweep:

```text
left -> right cycle: phase 0.2 = left-ish
right -> left cycle: phase 0.2 = right-ish
```

So the next test compares delayed credit using:

- raw phase;
- **oriented phase = phase bound to current sweep direction**;
- no phase;
- shuffled oriented phase;
- explicit oriented slot attacker;
- explicit raw slot attacker.

The question is:

> **Does the composed address become phase × sweep-direction rather than phase alone?**

Implementation: `experiments/fork_phase_direction_address.py`.
