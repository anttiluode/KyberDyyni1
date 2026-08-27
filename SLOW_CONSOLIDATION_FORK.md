# Slow consolidation under noisy partial fast search

Starting point: canonical `main` after Gate 8.

Gate 8 established a useful boundary:

> Fast structured probing is limited by the ratio of probe-induced consequence change to consequence noise.

That experiment deliberately froze slow structure. This fork reconnects the second half of the KyberDyyni architecture.

## Question

> **Can slow context memory extract the stable part of repeated noisy, incomplete fast corrections so that later related episodes require fewer scalar probes?**

The point is not merely to reduce late error. The stronger criterion is **probe amortization**.

A context recurs with a stable hidden correction of norm 0.60, but each encounter also contains a fresh nuisance correction of norm 0.08. The machine starts from its context-specific slow prior and may spend at most 512 scalar relevance measurements on a progressive Hadamard block-8 fast search.

The slow learner never sees the true target or true correction.

At the end of an encounter it may retain only:

```text
context
candidate = old slow prior + inferred fast correction
scalar delayed consequence
```

The consequence matures two encounters later.

## Attackers

- bounded delayed EMA-like structural memory;
- ordinary delayed EMA table;
- ordinary immediate EMA table;
- delayed running-mean / ridge table;
- delayed scalar-gain Kalman-style estimator;
- frozen slow memory;
- delayed credit applied to the wrong context.

The bounded learner is not expected to win by novelty. If ordinary EMA is numerically equivalent, that is an acceptable result: the architectural claim is the division of labor between fast temporary search and slow amortization.

## Metrics

The important measurements are:

- starting error before any fast probe;
- scalar probes actually consumed;
- true success after the fast episode;
- fraction of late episodes that need zero directional probes because the slow prior already lands inside the success radius;
- false halts from noisy scalar consequence;
- first-round versus late-round probe cost.

## Kill conditions

The fork fails if:

1. slow learning does not reduce future probe cost;
2. shuffled context credit works about as well as correctly addressed credit;
3. late improvement is only apparent because noisy stopping produces false success;
4. context memory memorizes the nuisance and degrades repeated performance.

If a boring EMA / estimator is best, use it.
