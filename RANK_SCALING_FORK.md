# Rank scaling of residual sign calibration

Gate 12 solved the four sign bits left by blind temporal alignment with ordinary scalar consequence.

Gate 13 attacks the obvious loophole: rank four is small.

## Scope

This fork deliberately **conditions on the Gate-11 temporal stage already having recovered axes and component identity**.

It does not rerun source separation.

The only unknown is one shared sign bit per transferable component. This isolates the scaling law of the final scalar-consequence calibration stage.

Ranks:

```text
4, 8, 16, 32
```

Consequence noise:

```text
0.02, 0.04
```

Context worlds:

- dense: every context excites every component;
- sparse4: only four random components are active per context;
- heavy-tail: all components exist, but later components are progressively weaker.

## Attackers

### Full bitwise

Probe +/- orientation for every component on every calibration context.

Cost:

```text
2R scalar measurements / context
```

### Top-4

Probe only the four components with the largest current coefficient magnitude.

Cost:

```text
<= 8 scalar measurements / context
```

### Active-4

Same fixed eight-measurement ceiling, but discount bits that have already been tested many times:

```text
priority_i = |coefficient_i| / sqrt(1 + probe_count_i)
```

This is deliberately mundane active measurement allocation.

### Random-4

Same fixed budget, random components.

### Shuffled full bitwise

Same expensive full measurement schedule, but half of the +/- evidence signs are randomly reversed.

## Metrics

The important distinction at high rank is between:

- literal fraction of sign bits correct;
- **energy-weighted** sign accuracy;
- actual transfer error/success;
- fraction of bits ever probed;
- scalar evaluations per latent rank.

A heavy-tail system may be useful even while weak irrelevant sign bits remain unresolved.

## Kill condition

If full bitwise remains cheap enough and the fixed-budget schemes lose badly, no active allocation role is earned.

If sparse or heavy-tail structure lets a tiny fixed probe budget approach full-bitwise transfer with much less measurement cost, then selective probing earns a narrow engineering role.

The test is not whether all sign bits can eventually be memorized. It is:

> **How much scalar consequence must be spent per reusable degree of freedom?**


## Result — selective consequence measurement scales much better than probing every bit

The rank attack does not kill the residual-calibration idea.

It changes what the useful resource is.

### Dense world

At rank 32, consequence noise 0.02:

```text
64 calibration contexts

method           transfer success    error       scalar evaluations

full bitwise          99.13%         0.0035            4096
top-4                 99.17%         0.0034             512
active-4              99.19%         0.0035             512
random-4              19.72%         0.3648             512
shuffled full          0.00%         0.8337            4096
```

At 128 contexts all three informed methods reach 100%, but full bitwise has spent 8192 scalar measurements while the four-bit methods have spent 1024.

That is an **8x measurement reduction** at rank 32.

At the harder consequence noise 0.04, selective measurement is not merely cheaper:

```text
rank 32, 128 contexts

full bitwise    92.47% success   8192 evals
top-4           98.36%           1024
active-4        96.08%           1024
```

Why can measuring fewer bits work better?

In a dense unit-norm context, individual component amplitudes shrink as rank grows. Full bitwise keeps adding low-SNR +/- evidence for tiny components. Selecting components that happen to be strongly expressed in the current context avoids spending measurements where the local consequence difference is mostly noise.

So the relevant scaling variable is not rank alone:

> **It is rank × excitation × consequence SNR.**

### Sparse-four world

The result becomes cleaner.

At rank 32 / noise 0.02 / 64 contexts:

```text
full bitwise    97.22% success   4096 evals
top-4           99.11%            512
active-4        98.92%            512
```

At 128 contexts both selective policies reach 100% with 1024 evaluations; full bitwise also reaches 100%, but costs 8192.

Here the reason is especially transparent. Twenty-eight components are exactly inactive on a given context. Probing them supplies **zero signal and nonzero measurement noise**.

The context itself tells the machine where consequence is informative.

### Heavy-tail world

This is where the difference between simple top-k and active allocation matters.

At rank 32 / noise 0.02 / 128 contexts:

```text
                    success   raw sign accuracy   energy-weighted sign   evals

full bitwise         92.0%          87.4%               99.18%           8192
top-4                66.6%          77.2%               97.82%           1024
active-4             91.8%          86.9%               99.19%           1024
```

The top-4 rule keeps revisiting the strongest components and neglects weak ones.

The tiny active rule:

```text
priority_i = |current coefficient_i| / sqrt(1 + times_probed_i)
```

eventually gives weaker under-tested components a turn.

At noise 0.04 the same pattern survives:

```text
rank 32, 128 contexts

full bitwise    79.7% success   8192 evals
top-4           59.0%           1024
active-4        81.7%           1024
```

Active-4 slightly **beats** the full O(R) attacker while using one eighth as many consequence measurements.

### A useful metric correction

Heavy-tail runs make raw sign accuracy misleading.

A rank-32 active map can have only ~82% of literal sign bits correct while already having ~98.5% **energy-weighted** sign accuracy.

The unresolved signs are mostly freedoms that barely contribute to the tested behavior.

So for a reusable control map, "all coordinates correct" is often the wrong success criterion.

> **Important degrees of freedom should be learned before weak degrees of freedom.**

That sounds obvious after the fact, but it is exactly what a finite consequence budget should do.

## What Gate 13 earns

Not a new learner.

The selected rule is almost embarrassingly simple.

What it earns is a resource-allocation role:

```text
current slow / latent coordinates
          |
          | which unresolved freedoms are
          | strongly expressed right now?
          v
small consequence-probe budget
          |
          v
global reusable orientation memory
```

At rank 32, four selected components per context can match full 32-component probing with roughly eight times fewer scalar measurements.

The strongest positive case is sparse excitation. The strongest boundary is weak dense excitation under high consequence noise.

The next unresolved loophole is upstream:

> **Gate 13 assumes the temporal separator already recovered all R axes and their component identities. Does *that* stage survive rank 8 / 16 / 32 when temporal signatures crowd together and the observation window is finite?**

That should be attacked before adding any more fast/slow machinery.
