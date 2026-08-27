# 2-D fork conclusion

This branch attacked the 1-D Gate-6 result in a genuinely two-dimensional uncertainty region.

## 1. Structured low-travel sampling survives, but the geometry is task-dependent

With equal continuous path budgets, structured trajectories beat matched-travel smooth random walk, while IID random remained the high-travel coverage upper bound.

Golden radial spokes were best for approximately centered directional uncertainty and cue loss. Spiral/Lissajous paths were better under systematic cue bias.

Therefore the 1-D result does not generalize as one privileged 2-D curve.

## 2. Binary miss-driven geometry switching fails

A fast EMA of hit/miss successfully detected trouble, but switching between radial and spiral/Lissajous paths usually reduced performance.

A miss says the current policy failed, not how the internal reference is wrong.

## 3. Relevant samples can calibrate the fast reference

Keeping the radial sampler fixed and using within-sweep relevance contrast to update an elastic 2-D correction vector produced the strongest general improvement:

```text
                         fixed radial    + fast contrast
RELIABLE                    90.0%            92.8%
MIXED                       61.0%            82.5%
SYSTEMATIC BIAS             36.8%            85.1%
LOSS hit fraction           78.5%            86.3%
LOSS reacquisition          41.7 ms          30.6 ms
```

No slow weights change.

The same local contrast rule on a smooth random walk is much worse.

Earned:

> **Structured local samples can calibrate an already-useful but imperfect reference through fast elastic state.**

## 4. Repeated fast calibration can become slow prior calibration

Six recurring contexts had stable cue-bias vectors and new target positions every encounter.

A successful temporary fast correction was delivered only after three later encounters.

```text
                              first       late
bounded delayed start error    0.435       0.231
bounded first-cycle hit        51.4%       82.6%

fast-only start error          0.435       0.429
fast-only first-cycle hit      51.4%       38.2%

shuffled-context start error   0.438       0.455
shuffled first-cycle hit       50.0%       41.7%
```

The explicit EMA attacker is numerically identical to the bounded learner because the budget never activates.

Earned:

> **Fast 2-D calibration can crystallize into delayed context-specific slow calibration without BPTT.**

Not earned:

> the current slow structural rule is better than an ordinary associative table.

## 5. Finite-difference attacker simplifies the sampler

A conventional center-crossing coordinate stencil nearly matches the radial path:

```text
                         radial     cardinal cross
RELIABLE                 92.8%          90.7%
MIXED                    82.5%          81.4%
SYSTEMATIC BIAS          85.1%          86.9%
LOSS reacquisition       30.6 ms        63.9 ms
```

The radial path is somewhat more robust to loss, but it is not uniquely required.

The important operation is therefore closer to:

> **structured local probing around a stable reference**

than to a particular theta-like trajectory.

## Consolidated artificial architecture

```text
             STABLE / SLOW REFERENCE
                       |
              fast approximate cue
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

The biological lineage helped expose the separation of roles, but the current digital implementation is increasingly conventional: structured zeroth-order probing + elastic state + delayed associative memory.

That is a useful outcome, not a failure.

## Next attacker

The clean next question is dimensional scaling.

In 2-D, +/-x and +/-y are cheap. In a D-dimensional latent space, a full coordinate stencil costs O(D) probes.

Attack:

- coordinate +/- basis;
- random orthogonal directions;
- Hadamard / structured sign probes;
- SPSA two-point probes;
- low-rank adaptive subspaces;
- smooth random motion;
- IID random proposals.

Question:

> **Can the fast calibration role survive high dimension without paying one probe pair per latent coordinate?**
