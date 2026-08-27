# Probe-width control fork

Starting point: the high-dimensional/noisy-consequence fork.

That fork found a real boundary:

```text
fast partial probe usefulness
    depends on
probe-induced consequence difference / consequence noise
```

A natural response was to make the spatial extent of the probe itself adaptive. This is also the cleanest computational analogue, in the current toy, of the experimentally observed modulation of theta-sweep width.

The question was deliberately narrow:

> **When scalar consequence is noisy, can the machine widen probes to raise measurement SNR and narrow them again when the local signal is strong, beating fixed probe widths under the same budget?**

## Experiment

Dimensions:

```text
128, 256
```

Worlds:

- dense hidden correction;
- sparse four-coordinate hidden correction.

Scalar relevance noise:

```text
sigma = 0.005, 0.01, 0.02, 0.04
```

Budget:

```text
512 scalar relevance evaluations
```

Every method uses the same progressive 8-direction Hadamard basis.

Attackers:

```text
fixed radius 0.10
fixed radius 0.20
fixed radius 0.28
fixed radius 0.40
fixed radius 0.60
simple coarse -> fine schedule
literal contrast/SNR adaptive radius
conservative capped contrast/SNR adaptive radius
```

The kill condition was written before seeing the result:

> If one fixed radius or a simple schedule matches or beats the adaptive controllers, sweep-width control has not earned an architectural role in this toy.

Implementation: `experiments/fork_probe_width_control.py`.

## Result — kill the adaptive-width mechanism here

The kill condition fires.

Across the harder 256-D conditions, **fixed radius 0.40 wins every tested noise level in both dense and sparse worlds**.

Selected mean final errors:

```text
D=256 dense

noise sigma       fixed .28     fixed .40     coarse->fine    adaptive capped
0.005               0.308         0.305          0.310            0.319
0.010               0.353         0.342          0.359            0.348
0.020               0.507         0.472          0.524            0.489
0.040               0.763         0.731          0.773            0.748
```

```text
D=256 sparse4

noise sigma       fixed .28     fixed .40     coarse->fine    adaptive capped
0.005               0.271         0.267          0.272            0.284
0.010               0.315         0.304          0.321            0.316
0.020               0.473         0.437          0.491            0.455
0.040               0.761         0.725          0.772            0.745
```

The literal rule — "low contrast means widen aggressively" — is substantially worse. At high noise it often drives the radius toward ~0.75 and amplifies the wrong lesson from noisy measurements.

The conservative adaptive rule gets much closer, and occasionally nearly ties the best fixed choice. But it does not beat the fixed-radius attacker reliably enough to justify another state variable or controller.

At 128-D the identity of the best fixed radius varies in a few conditions:

- dense sigma 0.005: fixed 0.40 wins;
- dense sigma 0.01: fixed 0.10 has the lowest mean error, although fixed 0.40 has a slightly higher threshold-success fraction;
- dense/sparse sigma 0.02: fixed 0.20 wins;
- dense/sparse sigma 0.04: fixed 0.40 wins;
- sparse sigma 0.005: fixed 0.60 and capped adaptation are essentially tied.

That variation is not enough. The adaptive controller was supposed to exploit it automatically, and it did not.

Receipt: `results/fork_probe_width_control_summary.json`.
CI source run: `33042024635`.

## Interpretation

This is useful pruning.

The current relevance function has one stationary characteristic scale:

```text
relevance(distance) = exp(-0.5 * (distance / 0.40)^2)
```

So it is not surprising that a probe radius around 0.40 is a strong general attacker. In this world, "adaptive attention width" mostly rediscovers — badly — a fixed scale already present in the consequence landscape.

Therefore do **not** add adaptive sweep width to the canonical KyberDyyni architecture on the strength of this experiment.

The neuroscience analogy should stay disciplined:

- the biological data motivate asking whether sampling geometry can be dynamically controlled;
- they do not imply that our toy needs a width controller;
- under the stationary relevance landscape tested here, it does not.

## What survived

The stronger result from the parent fork survives untouched:

```text
stable reference
+ partial structured probes
+ scalar relative consequence
+ finite fast budget
--------------------------------
can recover useful corrections
without a full coordinate gradient
```

but only while probe-induced consequence differences remain resolvable above measurement noise.

That is the current computational fact.

## Only defensible future width test

There is one reason to revisit width later: **if the consequence landscape itself changes scale**.

For example, randomize the unknown relevance width across episodes:

```text
0.15, 0.30, 0.60, 1.00
```

and deny the sampler that scale.

Then a single fixed radius cannot be pre-matched to the world. If an adaptive sampler learns the useful scale online under the same finite budget, width control would finally have a job.

Until such a changing-scale world is needed by the architecture, this fork is frozen as a negative result.
