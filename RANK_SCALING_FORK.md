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
