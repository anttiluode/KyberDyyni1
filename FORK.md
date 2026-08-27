# Control-law fork

This branch deliberately asks the Vollan/Ji control question repeatedly before choosing an engineering interpretation.

Canonical `main` remains the Gate-0→5 line. The fork keeps its own receipts so failed translations remain visible.

## Fork 1 — where should relevance act?

Same scalar fast relevance signal, no slow learning, three worlds.

Selected means:

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

Result: a stable upstream reference plus a movable downstream sampling coordinate is viable and is especially good in continuous pursuit. Random dither still wins raw search at enormous path travel.

Receipt: `results/fork_control_laws.json`.

## Fork 2 — mechanism map

One-factor Ji-like sensitivity map.

```text
theta period       frequency      mean cycle peak
130 ms               7.7 Hz          0.728 rad
115                  8.7             0.672
100                 10.0             0.593
 85                 11.8             0.490
 70                 14.3             0.121
```

Important result: **faster theta itself narrows the adaptation-generated sweep**. No reduction in adaptation or stronger anchor is required. At 70 ms organized alternation starts to degrade.

Other findings:

- adaptation is a strong mobility knob: mbar 6 gives almost no sweep; 12 gives ~0.59 rad; 18 gives ~1.21 rad and starts damaging alternation;
- stronger external tether narrows the sweep monotonically;
- stronger recurrence increases excursion modestly;
- larger theta-modulation amplitude broadens the sweep until a high-amplitude instability;
- ring tuning width has a non-monotonic effect.

Receipt: `results/fork_mechanism_map.json`.

## Fork 3 — separate stable reference and attention drive

Tested downstream coordinate translation, a second Gaussian attention drive, precision-weighted dual drive, partial release from reference, asymmetric gain, multiscale modules, prediction, phase reset and attackers.

Selected means:

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

Separate drives did not produce a general winner. The simple downstream coordinate remains the strongest attractor-based pursuit interpretation.

Asymmetric gain exposed a metric trap: it can make the **internal bias estimate** look accurate while actual sampled values remain poor. Future forks score both representation and useful sampling.

Receipt: `results/fork_attention_mechanisms.json`.

## Fork 4 — credit once per complete theta sweep

The previous task supplied relevance every millisecond, so faster theta did not create more complete sampling opportunities. Fork 4 allowed only **one fast-state update per complete theta cycle** while neural activation/adaptation remained continuous.

That did **not** rescue the scalar-reward interpretation.

```text
stationary acquisition:
  best continuous policy       ~1310 ms, 60% success
  no fast update                1600 ms,  0%

pursuit error:
  adaptive adaptation            1.319 rad
  downstream scale               1.382
  no fast update                 1.664
  random cycle dither            1.825

loss / reorientation:
  essentially every policy failed the 900-ms gate
  fixed 70-ms theta: 5% success; others ~0%
```

This is useful negative evidence:

> **More theta cycles per second are not enough if the sweep is also being asked to infer target direction from sparse scalar consequence.**

A secondary hint appears here: random dither loses badly once feedback is trajectory/cycle-integrated, while continuous policies retain some benefit. That is not yet a continuity result because overall task performance is poor.

Receipt: `results/fork_cycle_level_control.json`.

## Fork 5 — change the source of the control signal

The biological interpretation now needs a different attack.

A rat chasing visible bait may not require the EC/HPC sweep to discover bait direction from scalar reward. A visual salience signal, remembered target, or movement-plan vector may already provide **where** to attend. Vollan also reports internally generated movement-direction control and modulation during REM.

Fork 5 therefore gives the fast sampler a noisy directional cue computed outside the sweep. Slow structure remains frozen.

The upstream reference stays fixed. The downstream sampling axis follows the cue. Then we compare:

- empirical-foraging theta rate (~8.2 Hz);
- empirical-chasing theta rate (~9.1 Hz);
- confidence-interpolated empirical frequency;
- a wider 7.7→14.3 Hz sensitivity range;
- confidence-controlled adaptation;
- confidence-controlled external tether;
- confidence-controlled downstream sweep width;
- axis-only sampling;
- random samples around the cued axis;
- no directional cue.

Worlds:

1. reliable moving-target pursuit;
2. changing cue reliability;
3. repeated cue loss and reacquisition.

Metrics distinguish:

- axis accuracy;
- useful sampled value;
- fraction of cycles that actually hit the target region;
- utility per wall-clock second;
- cycle rate;
- sweep excursion;
- path travel;
- reacquisition latency.

Question:

> **Do the Vollan-like direction/width/frequency changes become useful when they are treated as an active-sensing policy driven by a separate control vector, rather than as a scalar-reward search algorithm?**

Implementation: `experiments/fork_directional_cue.py`.
