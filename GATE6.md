# Gate 6 — persistent structured trajectory and trajectory-frame address

Gate 6 consolidates the two post-Gate-5 research forks into the parts that survived direct attackers.

It is intentionally split into coverage and addressability.

## Question

> **Can a continuously evolving internal trajectory cover uncertainty efficiently enough to justify persistence, and can delayed consequence still refer to the correct event once the trajectory reverses?**

No claim here requires the Ji-like attractor. In fact, the digital winner is simpler.

## 6A — structured coverage, not generic continuity

The control-law fork first tested many artificial translations of Vollan-style sweep retuning.

The useful abstraction that survived was:

```text
stable reference
      +
fast approximate control vector
      +
structured sampler around uncertainty
```

A deterministic low-travel alternator approached IID-random target coverage while moving tens of times less distance. A smooth random walk at a comparable travel scale did not.

This killed the stronger claim that generic continuous dynamics were enough.

## 6B — remove the artificial cycle reset

The structured-sampling fork then matched continuous schedules by per-cycle path length.

If every schedule was forced to return to center, one-side alternation lost to bilateral and sorted low-discrepancy coverage.

The key correction was to preserve endpoint state across cycles.

```text
cycle n:      left ----------------------> right
cycle n+1:   right ----------------------> left
```

The shuttle therefore spends its travel budget covering the uncertainty interval instead of repeatedly paying a return-to-center cost.

12-seed selected receipt:

```text
RELIABLE CUE
IID random, unmatched          100.00% hit   travel 0.327 rad/ms
cross-cycle shuttle             99.17%       travel 0.0111
closed bilateral                98.33%

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

This earns a narrow geometric claim:

> **For a one-dimensional uncertainty interval, a persistent boundary-to-boundary shuttle is an extremely path-efficient coverage schedule.**

It does not establish that biological theta alternation exists for this reason.

## 6C — raw phase breaks when the trajectory reverses

A reversing shuttle changes the spatial meaning of raw phase:

```text
left -> right cycle: phase 0.2 = left-ish
right -> left cycle: phase 0.2 = right-ish
```

Six recurring contexts preferred different spatial positions among eight candidate events. Reward arrived four cycles later.

12-seed receipt:

```text
oriented phase                        95.54%
raw theta phase                       36.91%
explicit raw slot                     50.14%
no phase                              12.46%
shuffled oriented phase               12.49%
explicit oriented slot               100.00%
chance                                12.50%
```

The oriented phase is simply phase expressed in the current direction of traversal.

This refines Gate 2:

> **Delayed credit needs an event address in the coordinate frame of the currently unfolding trajectory.**

In this 1-D reversing task:

```text
event address ~= phase × sweep direction
```

An explicit oriented-slot address solves the task perfectly, so the architecture does not depend on phase being a biologically special code.

## Consolidated architecture

```text
               STABLE / SLOW REFERENCE
                         |
              fast control / uncertainty
                         |
                         v
              STRUCTURED FAST TRAJECTORY
                    <------------>
                         |
                  candidate events
                         |
              trajectory-local address
                         |
                    time passes
                         |
                 scalar consequence
                         |
                         v
                   SLOW STRUCTURE
```

Gate 5 established fast-state -> delayed slow prior.

Gate 6 establishes a useful structured fast trajectory and shows how a delayed address must transform with that trajectory.

The next missing composition is a realistic repeated-context task in which **coverage, trajectory address and slow prior all matter simultaneously**.

Before that, dimensionality is the cleanest attacker: the 1-D geometry may be doing most of the work.

## Canonical files

Coverage:

- `experiments/fork_cross_cycle_sequences.py`
- `results/fork_cross_cycle_sequences.json`

Address:

- `experiments/fork_phase_direction_address.py`
- `results/fork_phase_direction_address.json`

Exploratory histories:

- [FORK_CONCLUSION.md](FORK_CONCLUSION.md)
- [STRUCTURED_CONCLUSION.md](STRUCTURED_CONCLUSION.md)

## Still not earned

- the Ji attractor is necessary;
- theta phase is superior to explicit symbolic address;
- boundary shuttling generalizes to 2-D or higher dimensions;
- the trajectory-relative address has yet been coupled to Gate 5's slow prior in one end-to-end task;
- this beats standard active-search or attention systems on a realistic workload.
