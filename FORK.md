# Control-law fork

This branch deliberately asks the Vollan/Ji control question repeatedly before choosing an engineering interpretation.

Canonical `main` remains the Gate-0→5 line. The fork keeps its own receipts so failed translations remain visible.

## Fork 1 — where should relevance act?

Same scalar fast relevance signal, no slow learning, three worlds.

```text
                               stationary       pursuit error      reorientation
move generator anchor            113 steps        0.314 rad          66 steps
stable reference + bias          138              0.237             117
stable ref + explore/contract    127              0.211             100
theta-rate only                  272              0.971             259
width compression only           241              0.599             193
old naive focus                  266              0.802             234
random dither                     80              0.101              67
```

A stable upstream reference plus a movable downstream sampling coordinate is viable and especially good during continuous pursuit.

Receipt: `results/fork_control_laws.json`.

## Fork 2 — mechanism map

One-factor Ji-like sensitivity map:

```text
theta period       frequency      mean cycle peak
130 ms               7.7 Hz          0.728 rad
115                  8.7             0.672
100                 10.0             0.593
 85                 11.8             0.490
 70                 14.3             0.121
```

Faster theta alone narrows the adaptation-generated sweep. Adaptation is a strong mobility knob; stronger external tether narrows; theta-modulation amplitude broadens until instability.

Receipt: `results/fork_mechanism_map.json`.

## Fork 3 — separate stable reference and attention drive

Separate Gaussian drives, precision weighting, release from the reference, asymmetric gain, multiscale modules, prediction and phase reset did not produce a general winner. The simple downstream coordinate remained the strongest attractor-based pursuit interpretation.

Receipt: `results/fork_attention_mechanisms.json`.

## Fork 4 — credit once per complete theta sweep

Restricting fast-state updates to one per complete theta cycle did not rescue scalar-reward search.

```text
stationary: best policy ~1310 ms, 60% success
pursuit: adaptive adaptation 1.319 rad; no update 1.664
reorientation: essentially all policies fail 900-ms gate
```

This argues that the sampler probably should not be forced to infer target direction from sparse scalar consequence.

Receipt: `results/fork_cycle_level_control.json`.

## Fork 5 — give the fast sampler a direction / attention vector

The next interpretation supplies a noisy directional cue outside the sweep, as an analogue of visual salience or an internally generated movement-plan vector.

That changes the picture sharply.

### Reliable moving-target pursuit

```text
                                  target-hit cycles    utility/sec    path travel
random around cued axis               100.0%             10.00        0.400
wide frequency adaptation              99.0%             11.93        0.0127
adaptive adaptation                     98.3%              9.99        0.0120
adaptive downstream width               97.5%              9.99        0.0119
chasing-like ~9.1 Hz                    94.3%              9.07        0.0132
foraging-like ~8.2 Hz                   86.3%              8.16        0.0129
axis only                               87.7%              9.96        0.0016
no directional cue                      17.3%              5.96        0.0118
```

Important: `axis_only` has extremely high mean value when the noisy cue is accurate, but **fewer cycles actually hit the true target region**. The sweep hedges directional cue error.

### Changing cue reliability

```text
random around axis                       88.1% hit cycles
adaptive downstream width               77.1%
adaptive adaptation                      76.9%
adaptive tether                          75.4%
wide frequency adaptation                75.2%
axis only                                55.2%
```

The continuous adaptive policies give up some target-hit probability relative to IID random proposals, but at roughly 30× lower path travel.

### Cue loss and return

```text
foraging-like broad/slow               ~25 ms reacquisition
adaptive adaptation                    ~29 ms
adaptive tether                        ~29 ms
adaptive downstream width              ~29 ms
wide frequency adaptation              ~31 ms
chasing-like faster/narrower           ~37 ms
axis only                              ~113 ms
no directional cue                    ~404 ms
```

This is the first fork that gives the broad↔focused distinction a clean computational role:

> **When a direction cue is reliable, focused/high-rate sampling concentrates effort and tracks well. After cue loss, broader sampling reacquires sooner.**

This is compatible with the active-sensing interpretation rather than the scalar-optimizer interpretation.

Receipt: `results/fork_directional_cue.json`.

## Fork 6 — attack the attractive result

Do not stop at the first biologically pleasing story.

Fork 6 asks whether the Ji-like population dynamics are actually needed once we have discovered the useful abstraction.

Confidence now controls both empirical-like frequency and angular spread. Compare:

- Ji attractor: confidence controls frequency + downstream width;
- Ji frequency only;
- Ji width only;
- Ji adaptation switch;
- **engineered speed-matched alternating sweep** with no attractor;
- **smooth bounded random walk** with similar step scale and no theta oscillator;
- IID random samples;
- axis only.

The cheap engineered sweep is intentionally strong: it moves a single continuous state toward alternating side targets with a 0.013-rad/ms speed limit, close to the observed path-travel scale of the Ji sampler.

Question:

> **Is the useful thing the Ji/adaptation mechanism, or simply a continuously moving confidence-controlled sampling policy?**

Implementation: `experiments/fork_mode_switch_attack.py`.
