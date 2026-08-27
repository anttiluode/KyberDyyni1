# Structured sampling fork

Starting point: frozen `control-law-fork` conclusion `7f086f0`.

The previous branch ended with:

> structured low-travel sampling around an uncertain fast control vector

This fork asked whether left/right alternation itself was special, then whether the resulting continuous trajectory could compose with delayed local credit.

## Fork 1 — equal-path closed-cycle schedules

All continuous schedules received equal wall time, 100 probes per 100-ms cycle, the same noisy directional cue, no slow learning, and exactly the same per-cycle total-variation budget.

```text
RELIABLE
IID random, unmatched          100.0%   travel 0.327
bilateral triangle              98.3%   travel 0.0113
bilateral sine                  97.5%   travel 0.0113
sorted VdC coverage             97.2%   travel 0.0113
one-side alternating            92.5%   travel 0.0113

MIXED RELIABILITY
bilateral sine                  77.8%
sorted VdC                      76.7%
bilateral triangle              76.5%
one-side alternating            71.5%
```

Result:

> **One-side-per-cycle alternation is not special if every cycle is artificially forced to return to center.**

Receipt: `results/fork_structured_sequences.json`.

## Fork 2 — preserve state across cycle boundaries

The hidden artificial cost in Fork 1 was the return to center.

Let the endpoint persist:

```text
cycle n:      left ----------------------> right
cycle n+1:   right ----------------------> left
```

Selected results:

```text
RELIABLE
IID random                      100.0%   travel 0.327
cross-cycle shuttle              99.17%  travel 0.0111
closed bilateral                 98.33%

MIXED RELIABILITY
IID random                       90.56%
cross-cycle shuttle              89.31%
closed bilateral                 78.33%

SYSTEMATIC CUE BIAS
IID random                       99.58%
cross-cycle shuttle              95.69%
closed bilateral                 73.89%

LOSS / RETURN
IID random                        0 ms
cross-cycle shuttle               5.6 ms
closed one-side                  19.4 ms
closed bilateral                 27.8 ms
axis only                        80.6 ms
smooth random walk              227.8 ms
```

Result:

> **Boundary-to-boundary alternation becomes extremely path-efficient once the sampler is genuinely continuous across cycles.**

It nearly matches IID random coverage while moving roughly thirty times less distance.

Receipt: `results/fork_cross_cycle_sequences.json`.

## Fork 3 — phase must live in the trajectory's coordinate frame

A reversing sweep breaks raw phase as a stable spatial address:

```text
left -> right: phase 0.2 = left-ish
right -> left: phase 0.2 = right-ish
```

Delayed reward arrived four cycles later. Six recurring contexts preferred different spatial positions among eight within-sweep candidates.

```text
oriented phase
(phase bound to sweep direction)       95.54%

raw theta phase                        36.91%

explicit raw slot                      50.14%

no phase                               12.46%
shuffled oriented phase                12.49%

explicit oriented slot                100.00%
chance                                 12.50%
```

The explicit slot attacker still wins perfectly, so there is no claim that phase is uniquely privileged.

The new earned statement is:

> **For a reversing continuous trajectory, delayed local credit needs an address in the trajectory's current coordinate frame. Phase × sweep-direction is sufficient; raw phase is not.**

This refines Gate 2 rather than merely repeating it.

Receipt: `results/fork_phase_direction_address.json`.

## What survived

The fork now gives three connected computational pieces:

```text
stable / slow reference
        |
fast directional uncertainty
        |
        v
continuous boundary-to-boundary shuttle
        |
candidate events along trajectory
        |
oriented phase / local trajectory address
        |
        ... delayed consequence ...
        |
slow context-dependent preference
```

The strongest result is no longer "the brain alternates left/right, so copy it."

It is:

> **If a sampler has to cover a one-dimensional uncertainty interval continuously, carrying its endpoint into the next sweep makes boundary-to-boundary alternation a path-efficient coverage strategy; if delayed consequences must refer back to events on that reversing path, the event address must include the path's orientation.**

## Still not earned

- the Ji attractor is required;
- theta phase is superior to an explicit symbolic coordinate;
- this geometry remains optimal in higher-dimensional latent spaces;
- the learned phase preference is yet the same thing as Gate 5's slow prior that changes where future search begins;
- a realistic AI workload benefits from this decomposition.

Those are later attacks, not conclusions to smuggle in.
