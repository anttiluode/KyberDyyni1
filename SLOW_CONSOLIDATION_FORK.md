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


## Fork 1 result — slow learning worked, but the old fast controller fought it

The first result is more interesting than a clean pass.

Correctly addressed slow memories learned a large fraction of the stable context correction. At dense 128-D / consequence noise 0.01:

```text
                         late start error    late probes    late success
EMA delayed                    0.238             503            1.7%
Kalman delayed                 0.210             490            2.5%
frozen                         0.606             429           31.7%
shuffled context               0.695             507            0.8%
```

So the slow memory clearly learned something: start error fell from about 0.61 to 0.21--0.24, while shuffled credit made the prior worse.

But that **did not amortize the search**. The legacy fast stage had been tuned for a residual error around 0.60. Once slow memory delivered the state near the target, the same fixed probe radius and step size became destructive. It often moved a good prior away from the solution.

At noise 0.02 the same interaction is even easier to see. The Kalman attacker begins late episodes at about 0.174 error, with 75% of starts already inside the 0.18 success radius, yet final true success falls to 43%. The fast stage is erasing part of the slow stage's work.

That means Fork 1 does **not** support "slow memory failed." It supports:

> **A fast/slow architecture needs a handoff rule. An exploration controller that is useful far from the solution can become harmful after slow memory has already done most of the work.**

This is a standard optimizer problem before it is a neuroscience problem.

## Fork 2 — ordinary trust-region / accept-reject attack

The next attack therefore changed only the fast residual controller.

Two conventional safeguards were added:

1. infer a rough residual distance from the noisy known relevance curve and shrink probe radius / step size near high relevance;
2. spend one extra scalar measurement on the proposed update and reject it when measured relevance does not improve.

No new slow-learning mechanism was added.

### Dense 128-D, noise 0.01

```text
                              late start   late final   late probes   success
Kalman + scaled/accept          0.164        0.164          30.7      95.8%
EMA + scaled/accept             0.162        0.162          24.2      93.3%
frozen + scaled/accept          0.606        0.177         411.3      63.3%
shuffled + scaled/accept        0.666        0.191         463.3      38.3%
```

The learned systems now save about **92--94%** of the scalar probe cost relative to their own first-round cost. About 96% of late dense episodes begin inside the target radius before any directional probe.

### Sparse4 128-D, noise 0.01

```text
                              late start   late final   late probes   success
Kalman + scaled/accept          0.164        0.164          32.5      94.2%
EMA + scaled/accept             0.162        0.163          39.9      90.8%
frozen + scaled/accept          0.605        0.160         345.5      80.8%
shuffled + scaled/accept        0.667        0.189         415.9      50.0%
```

Again the point is not that Kalman is special. A boring EMA is almost as good and is actually cheaper in the dense 0.01 condition.

### Harder consequence noise 0.02

The benefit survives, but noisy stopping becomes the new boundary.

```text
dense:
Kalman + scaled/accept      145.6 probes   58.3% final success   71.2% savings
EMA + scaled/accept         187.5          49.2%                 62.9%
frozen                      505.0           0.0%
shuffled                    505.0           0.0%

sparse4:
Kalman + scaled/accept      144.1 probes   58.3%                 71.5% savings
EMA + scaled/accept         189.4          42.5%                 62.5%
frozen                      499.7           0.0%
shuffled                    505.0           0.0%
```

The remaining failure is visible in the gap between **zero-directional-probe success** and final success. With noise 0.02, many late starts are already good, but noisy scalar thresholding sometimes tells the controller to keep searching. Even cautious search can then lose some of those good states.

## What Gate 9 earns

The strongest surviving statement is deliberately ordinary:

> **Repeated noisy fast corrections can be amortized into context-specific slow priors. Once the fast residual optimizer is made cautious near a learned solution, later related episodes can require an order of magnitude fewer scalar probes.**

And two pieces are now quite hard to dismiss as bookkeeping:

- **context address matters** — shuffled delayed credit reliably destroys the benefit;
- **fast/slow handoff matters** — the naive fast stage can erase a good slow prior.

What does *not* earn novelty:

- the slow update rule. EMA and Kalman-style estimation are sufficient;
- the trust-region handoff. Conventional distance scaling and stochastic accept/reject are sufficient.

That is still a useful architectural result. The machine is becoming:

```text
context / stable reference
          |
          v
   slow learned prior
          |
          v
 cautious fast residual probes
          |
          v
 temporary correction
          |
          v
 delayed addressed consolidation
```

The next attack should not invent a fancier learner. It should attack **transfer**: does the learned slow information remain useful when nuisance coordinates or the latent basis change, or is this merely an explicit context lookup table learning one coordinate system?
