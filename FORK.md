# Control-law fork

This branch asks the Vollan-inspired focused-sampling question **many different ways before selecting an engineering interpretation**.

Canonical `main` remains the Gate-0→5 line. This branch is deliberately exploratory.

## Fork 1 — control-law matrix

All controllers used the same scalar fast relevance signal and **no slow learning**.

The first bakeoff separated:

- moving the generator's external anchor;
- keeping the upstream reference fixed while translating a downstream sampling coordinate;
- theta-rate modulation only;
- downstream angular-sector compression only;
- coupled rate/width under an approximate fixed sampling budget;
- broad-while-uncertain / contract-when-confident sampling;
- downstream relevance gating;
- one-side suppression;
- learned phase-priority weighting;
- the original naive adaptation+tether+rate translation;
- random dither;
- no-relevance control.

Every controller was evaluated in:

1. stationary target acquisition;
2. continuously moving pursuit;
3. target loss followed by abrupt reorientation.

### First result

Selected mean metrics across 6 seeds:

```text
                               stationary       pursuit error      reorientation
move generator anchor            113 steps        0.314 rad          66 steps
stable reference + bias          138              0.237             117
stable ref + explore/contract    127              0.211             100
theta-rate only                  272              0.971             259
width compression only           241              0.599             193
rate+width fixed budget          364              1.068             277
old naive focus                  266              0.802             234
random dither                     80              0.101              67
no relevance                     483              1.733             290
```

The important qualitative fork is:

> **A stable upstream reference plus a movable downstream sampling coordinate works, and in continuous pursuit it outperforms moving the generator anchor.**

A second useful hint is:

> **Broad while uncertain, contract when confident** improves the stable-reference controller in all three worlds.

The direct "make theta faster / narrow the sweep" translations are poor on these tasks. Random dither still wins raw search, at vastly higher probe travel.

Canonical receipt: `results/fork_control_laws.json`.

## Fork 2 — mechanism map

Before guessing another control law, vary Ji-like mechanism parameters one at a time and measure what they actually change:

- adaptation strength;
- theta period;
- theta modulation amplitude;
- external tether gain;
- recurrent gain;
- ring tuning width.

Measure separately:

- theta-cycle peak excursion;
- alternation;
- stepwise continuity;
- bump coherence;
- actual cycle rate.

The point is to stop mapping one population-level Vollan observation onto one cellular parameter by intuition.

## Fork 3 — architectural translations

Test several ways to keep a **stable reference** while steering an elastic internal sampler:

- a separate transient Gaussian attention drive;
- confidence/precision-weighted attention drive;
- partial release from the stable reference plus stronger attention drive;
- broad asymmetric gain rather than a second attractor input;
- multiscale scanner modules selected by confidence;
- predictive downstream bias for moving targets;
- phase reset on target return;
- downstream explore/contract as the current best stable-reference baseline.

The key question is no longer merely "can focus change width/frequency?"

It is:

> **Where in the architecture should relevance act so that a stable reference can coexist with flexible internal sampling?**

The dedicated GitHub Actions workflow computes all three forks on this branch.
