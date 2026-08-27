# Gate 5 result — fast relevance, delayed consequence, slow prior

Gate 5 tested the architectural claim that motivated KyberDyyni:

> **Can one continuously running machine react to relevance immediately through fast state, then consolidate successful fast searches into bounded slow structure that improves later search—without BPTT?**

Implementation: `experiments/gate5_fast_slow_memory.py`

Canonical receipt: `results/gate5_fast_slow_memory.json`

## World

There are six recurring contexts. Each context has a hidden target distribution on a circular latent manifold.

On each encounter the system receives the context but not the target location. It samples its own internal probes and receives only scalar relevance:

```text
value = 0.5 + 0.5 cos(probe - hidden_target)
```

The adaptation attractor uses 64 cells. Its population state is not reset between probe steps. On context changes, the temporary fast offset and focus state decay rather than being used as persistent memory.

Eight independent seeds were evaluated; each seed contains six rounds through the six contexts, with 700 probe steps per encounter.

## Gate 5A — fast temporary computation

Slow structure is frozen.

The fast directional state uses the relation between the self-generated probe displacement and whether that probe was better or worse than a running relevance baseline:

```text
fast_offset += eta * (value - baseline) * (probe - working_anchor)
```

No slow weight changes.

Results:

```text
                                  first acquisition   success   mean value   travel/step
axis-only fast state                  92.6 ± 55.4      1.000      0.921       0.0146
axis + naive focus                   334.1 ±111.0      0.646      0.884       0.0049
no fast relevance                    700.0 ±  0.0      0.000      0.503       0.0118
```

### Earned

> **A fast continuously evolving state can perform useful temporary computation while slow structure remains exactly unchanged.**

### Unexpected boundary

The naive focus mechanism deliberately borrowed the Vollan-shaped idea of faster/narrower sampling: as relevance rises, theta speeds up, adaptation is reduced, and the sensory tether strengthens.

It **reduces internal travel** but hurts acquisition badly on this task.

That is useful negative evidence. The biological observation that pursuit narrows and speeds theta sweeps does not imply that our simplistic parameter translation is a good generic optimizer.

## Gate 5B — delayed crystallization

At the end of an encounter, the best self-generated probe is retained as a small local packet:

```text
(context, candidate_angle, scalar_consequence)
```

The consequence is delivered only after two later encounters.

When it matures, the bounded slow learner performs a local circular update and projects the complete slow structure back into a finite L1 budget.

No intervening fast-state trajectory is stored for reverse differentiation.

Results:

```text
                                  first acq   late acq   late start error   slow capacity
bounded delayed slow prior           92.6       37.1        0.082 rad          5.73
slow frozen                           92.6       87.1        1.560 rad          0
context credit shuffled               80.0      109.7        1.760 rad          2.14
explicit EMA table                    92.6       45.4        0.083 rad          6.93
engineered sweep + bounded prior     102.7       12.9        0.082 rad          5.73
random dither + bounded prior         94.1        1.6        0.090 rad          5.71
```

Late success is 100% for all correctly addressed learning systems above.

### Earned

> **Delayed local consequence can crystallize a successful fast computation into slow structure that improves later behavior.**

The key causal control is the shuffled-context condition. With the same number of slow updates but the delayed packets applied to random contexts, the learned starting error is worse than the frozen system and late acquisition slows to about 110 steps.

Persistent addressability matters.

### Attacker result

The explicit EMA table reaches essentially the same learned starting error as the bounded structural prior.

The engineered sweep and random dither use the same slow learner and outperform the adaptation scanner on raw acquisition speed.

So Gate 5 supports the **fast-state / slow-memory decomposition**, not the necessity of the Ji-like attractor.

Random dither still has a major continuity cost:

```text
adaptation scanner     0.0135 rad travel / step
random dither          0.3996 rad travel / step
```

That is now the motivation for Gate 6: make path continuity itself consequential.

## Where phase is—and is not

Gate 5 does not require phase. Its delayed packet explicitly retains context plus candidate angle.

That is intentional scientific bookkeeping.

Gate 2 separately established that theta phase can serve as an address when multiple otherwise-identical candidate events occur within one sweep and consequence arrives later.

The stronger combined system remains untested:

```text
self-generated attractor sweep
        |
multiple candidate events
        |
phase/context-local eligibility
        |
later consequence
        |
slow prior
```

Do not cite Gate 5 as evidence for that combined claim.

## Answer to Gate 5

**Yes, with an important qualifier.**

Fast state can solve the immediate task without slow learning. Delayed, correctly addressed local consequence can later make that solution available as slow prior knowledge, with no BPTT.

But nothing in Gate 5 says that the biological-looking attractor is necessary. On this simple latent-search world, cheap proposal mechanisms are better searchers.

The next useful attack is therefore not “add more hippocampus.” It is:

> **Does continuous internal trajectory become advantageous when transitions themselves carry state, cost, hysteresis, eligibility, or causal consequences?**
