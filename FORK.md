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

Supplying a noisy directional cue outside the sweep changes the picture sharply.

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

The sweep hedges directional-cue error: axis-only stares close to the noisy point estimate, but fewer cycles actually hit the true target region.

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

Receipt: `results/fork_directional_cue.json`.

## Fork 6 — attack the attractive result

Confidence controls angular spread/rate, but now the Ji attractor competes with cheap continuous samplers.

### Reliable cue

```text
                                  target-hit cycles    path travel
IID random                             100.0%             0.324
engineered speed-matched               99.1%             0.0100
Ji adaptation switch                   98.1%             0.0108
Ji width switch                        96.9%             0.0100
Ji frequency + width                   91.7%             0.0100
axis only                              84.5%             0.0015
smooth random walk                     62.0%             0.0140
```

### Mixed cue reliability

```text
IID random                              85.8%
engineered speed-matched               83.9%
Ji adaptation switch                   75.6%
Ji width switch                        72.9%
Ji frequency + width                   66.3%
axis only                              51.2%
smooth random walk                     47.4%
```

### Cue loss / return

```text
engineered speed-matched                0 ms reacquisition
IID random                              0 ms
Ji width / adaptation                  29 ms
Ji frequency controls                  34 ms
axis only                             135 ms
smooth random walk                    208 ms
```

The strongest result of the fork is therefore **not** that the Ji-like attractor is required.

A trivial deterministic continuous alternator nearly matches IID random target coverage while moving about **30× less distance**, and it beats the Ji variants in the mixed/loss worlds.

But the matched-travel smooth random walk performs badly.

So the surviving abstraction is narrower:

> **Structured alternating coverage matters; generic continuity does not.**

The Ji dynamics are a biologically plausible way to *generate* such a scanner. They are not currently the best digital implementation of the operation.

Receipt: `results/fork_mode_switch_attack.json`.

## What survived the six forks

1. **Stable reference + elastic sampler** is a useful decomposition.
2. Relevance should often act on a **downstream sampling coordinate**, not overwrite the stable reference.
3. Faster theta changes sweep dynamics by itself; do not automatically translate population observations into adaptation/tether changes.
4. Sparse scalar consequence is a poor source for immediate sampling direction in these tasks.
5. A separate fast directional/control signal makes the sweep useful as **active sensing**.
6. Broad sampling helps after uncertainty/loss; focused/high-rate sampling helps when the cue is reliable.
7. IID random proposals remain a brutal coverage attacker but pay enormous path-travel cost.
8. **Deterministic structured alternation** captures most of that coverage at low travel.
9. A smooth random walk with similar travel does not, so the result is not “continuity alone.”
10. The Ji attractor remains biologically interesting but is not currently necessary for the digital architecture.

The next research fork, if pursued, should attack **structured sampling sequences themselves**: alternating sweeps versus other low-discrepancy / deterministic coverage schedules under equal path, compute, and delayed-credit budgets.
