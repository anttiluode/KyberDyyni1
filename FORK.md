# Control-law fork

This branch deliberately asks the Vollan/Ji control question repeatedly before choosing an engineering interpretation.

Canonical `main` remains the Gate-0→5 line. Results here are experimental fork receipts.

## Fork 1 — where should relevance act?

Same scalar fast relevance signal, no slow learning, three worlds: stationary search, moving pursuit, loss/reorientation.

Selected 6-seed means:

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

First fork result:

> **A stable upstream reference plus a movable downstream sampling coordinate is viable and is better than moving the generator anchor during continuous pursuit.**

Broad-while-uncertain / contract-when-confident was the best stable-reference attractor controller across the three first worlds.

Receipt: `results/fork_control_laws.json`.

## Fork 2 — which Ji-like knobs actually move which observables?

One-factor mechanism map, 4 seeds per setting.

### Adaptation

```text
mbar       mean cycle peak     alternation
 6             0.004 rad          0.543
 9             0.188              1.000
12             0.593              1.000
15             0.781              1.000
18             1.206              0.913
```

Adaptation is a strong mobility knob, but excessive adaptation starts to damage clean alternation.

### Theta period

```text
period       Hz          mean cycle peak      alternation
130 ms       7.7             0.728 rad          1.000
115          8.7             0.672              1.000
100         10.0             0.593              1.000
 85         11.8             0.490              1.000
 70         14.3             0.121              0.841
```

This is the important surprise:

> **Faster theta alone narrows the adaptation-generated sweep.**

No reduction in adaptation and no stronger anchor are required. At sufficiently high frequency the organized alternation itself begins to degrade.

### Other sensitivities

- stronger external tether monotonically narrows the sweep (0.659 → 0.440 rad over the tested range);
- stronger recurrence increases excursion modestly while preserving coherence;
- theta-modulation amplitude strongly increases excursion until a high-amplitude instability;
- ring tuning width is non-monotonic.

Receipt: `results/fork_mechanism_map.json`.

## Fork 3 — architectural steering with a stable reference

The next bakeoff asked whether a fixed reference can coexist with a separate elastic control pathway.

Tested:

- downstream explore/contract coordinate;
- separate Gaussian attention drive;
- precision-weighted attention drive;
- partial release from the reference plus attention drive;
- broad asymmetric gain field;
- two adaptation modules at different scales;
- predictive downstream bias;
- phase reset on target return;
- moving-anchor and random attackers.

Selected 5-seed results:

```text
                                  stationary     pursuit error     reorientation
random dither                        81             0.096              67
move anchor                         112             0.310              62
downstream explore/contract         124             0.208             100
dual-drive release                  116             0.659              62
dual-drive precision                152             0.704              63
asymmetric gain                      95             0.734              53
multiscale modules                  300             0.547             241
predictive downstream               136             0.279             180
```

The separate-drive ideas did not produce a general winner. The simplest downstream coordinate remains the best attractor-based pursuit mechanism.

A warning from this fork: asymmetric gain looks excellent if we measure **internal bias acquisition**, but its mean sampled value is poor. It can move an internal estimate without actually putting useful probes on the target. Future gates must score both internal state and what the sampler physically represents.

Receipt: `results/fork_attention_mechanisms.json`.

## Fork 4 — ask theta frequency the right computational question

The first control-law task gave relevance feedback every **millisecond**. That means faster theta did not generate more complete sampling opportunities per wall-clock second; it merely changed the oscillator while the learner already saw feedback continuously.

That is not the natural interpretation of "high-rate theta sampling."

The next experiment therefore updates the fast controller **once per complete theta sweep**:

```text
population state persists
        |
complete theta cycle
        |
integrate relevance along the sweep
        |
one fast-state update
        |
next cycle
```

Same wall-clock time, no slow learning, stable upstream reference.

It compares several different mechanisms that all can create a broad→narrow transition:

- fixed 70 / 100 / 130 ms theta periods;
- confidence-controlled theta period;
- the inverse frequency law;
- confidence-controlled external tether;
- confidence-controlled adaptation;
- confidence-controlled theta amplitude;
- downstream sweep scaling;
- random cycle dither;
- no fast update.

Question:

> **Does slow/broad while uncertain → fast/narrow when confident become useful once theta frequency actually changes the number of complete internal sweeps per unit time?**

This is `experiments/fork_cycle_level_control.py`.
