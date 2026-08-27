# Gate 5 plan — fast relevance, delayed consequence, slow prior

Gate 4 established that the adaptation-generated sweep can perform derivative-free internal search, but it also established an attacker boundary: engineered sweeps and random dither acquire the simple target faster.

Gate 5 should therefore test the architectural claim that originally motivated KyberDyyni rather than another generic optimizer benchmark.

> **Can one continuously running machine react to relevance immediately through fast state, then consolidate repeated successful fast searches into bounded slow structure that improves later search?**

## World

Use several recurring contexts. Each context has a hidden target distribution on the circular latent manifold rather than one permanently fixed target.

On a first encounter, the machine receives the context but does not know the target location. It must search using only scalar values at self-generated probes.

During the encounter, relevance is available immediately. Final consequence arrives later.

The scanner is never reset between probe steps. Context switches should not reset its internal dynamical state either unless a control experiment explicitly does so.

## Gate 5A — fast relevance modulation

Slow structure is frozen.

A relevance signal may alter only fast state/control. Test whether it can change:

- scan direction / anchor bias;
- effective sweep sector or width;
- sampling frequency / theta period;
- exploitation vs exploration behavior.

The pass condition is not merely that parameters change. The modulation must improve an online objective under a matched probe-travel or sample budget.

Required comparisons:

```text
adaptation scanner + fast relevance
adaptation scanner, fixed policy
engineered theta scanner + fast relevance
random dither with matched travel/sample budget
```

If the adaptation scanner needs an externally scripted control that gives no advantage over `ThetaScanner`, the biological mechanism has not earned anything here.

## Gate 5B — delayed consolidation

After the fast search has produced candidate events, final consequence arrives after a delay.

Use phase/context-local retained traces to update a bounded slow prior. The slow prior should encode only information available through the machine's own successful searches; do not hand it target labels.

On a later recurrence of the same context, remove the original fast relevance cue and measure whether the slow prior improves the *starting search policy*.

The important before/after comparison is:

```text
first encounter:
  useful fast behavior can appear before slow learning

later encounter:
  useful bias is present before the fast search rediscovers it
```

## Attackers

At minimum:

- frozen slow structure;
- shuffled context-to-slow-prior mapping;
- explicit slot/counter addressing where relevant;
- engineered theta sweep + the same slow learner;
- random dither + the same slow learner;
- ordinary context -> target exponential moving average;
- ordinary contextual bandit / table when the context set is finite.

The EMA/table attacker is especially important. If a tiny explicit context prior matches or beats KyberDyyni with less state and no phase machinery, that is the product answer.

## Metrics

Report at least:

- first-encounter acquisition steps;
- repeated-context acquisition steps;
- improvement from first to repeated encounter;
- success fraction;
- tracking error;
- probe travel per step;
- samples used;
- slow capacity used;
- performance with slow weights frozen;
- performance after context-prior shuffling;
- retention after intervening contexts.

Do not report only final reward.

## Pass / kill interpretation

A strong result would be:

1. fast relevance helps on the first encounter while slow weights remain unchanged;
2. delayed consequence later changes bounded slow structure;
3. on recurrence, the system starts with a measurably better prior before receiving the old fast cue;
4. phase/context shuffling damages that benefit;
5. the benefit survives ordinary attacker comparisons or exposes a specific tradeoff such as continuity, storage, or delayed-address robustness.

A weak but useful result is that an ordinary EMA/table wins. That would tell us that fast-slow separation is useful but theta/phase machinery is unnecessary for this task.

Gate 5 is where the architecture should either start becoming a machine or start getting pruned.
