# High-dimensional probe fork

Starting point: consolidated Gate-7 `main` at `1326aa5`.

Gate 7 simplified the fast sampler to a conventional-looking operation:

```text
stable reference
      +
small structured local probe basis
      +
relative relevance
      ->
fast correction
```

In 2-D, a +/-x, +/-y stencil is cheap. That does not imply the mechanism scales.

## Fork 1 — fixed probe budget versus latent dimension

Dimensions:

```text
2, 4, 8, 16, 32, 64
```

Two bias regimes:

- dense correction spread across all dimensions;
- sparse correction occupying four unknown coordinates.

Every trial starts with a cue error of about 0.6 in Euclidean norm. No slow weights change.

Attackers:

- full coordinate +/- basis — deliberately expensive O(D) upper bound;
- 8-coordinate block, cycling across dimensions;
- 8 Hadamard directions;
- 8 random orthogonal directions;
- 8 Rademacher sign directions;
- 8 normalized Gaussian directions;
- SPSA-style one random sign direction / two probes;
- point estimate / no fast probing.

Fixed-block methods get at most:

```text
16 directional probes + one center relevance probe per cycle
```

regardless of dimension.

The full coordinate attacker receives `2D + 1` probes per cycle.

Each paired probe supplies only scalar relevance. The fast update follows the estimated correction direction; it receives no true gradient.

Metrics include:

- error after 1 and 4 cycles;
- final error after 10 cycles;
- success under 0.18 error;
- cycles and total probes to success;
- cosine between estimated and true correction direction;
- path used by the probe sequence.

Question:

> **Can the Gate-7 fast-calibration role survive increasing latent dimension without paying one probe pair per coordinate?**


## Fork 1 result — fixed directions eventually hit a horizon

The first run held the number of fast update cycles at 10. Full coordinates were allowed their natural `2D+1` scalar evaluations per cycle, while block methods used at most 17.

Dense-bias selected means:

```text
D=16
full coordinate        100% success   ~118 probes to success
coordinate block8      100%           ~92
Hadamard block8        100%           ~94
random orthogonal8     100%           ~98

D=32
full coordinate        100%           ~238
coordinate block8       91.7%         ~135
Hadamard block8         91.7%         ~136
random orthogonal8      62.5%         ~165

D=64
full coordinate        100%           ~478
coordinate block8       12.5%         ~169 max
Hadamard block8         20.8%         ~169 max
random orthogonal8       0%
```

Sparse-four-coordinate bias gives Hadamard mixing a clearer advantage at D=32:

```text
full coordinate        100% success   ~233 probes
Hadamard block8        100%           ~125
coordinate block8       33.3%         ~165
```

At D=64, however, even Hadamard mostly fails inside 10 cycles.

SPSA's two directional probes collapse much earlier: its gradient-direction cosine falls from ~0.79 at D=2 to ~0.09 at dense D=64.

This does **not** yet prove an O(D) lower bound. The fixed-block methods were simply denied the hundreds of scalar evaluations that the full-coordinate attacker consumed.

Receipt: `results/fork_highdim_probe_scaling.json`.

## Fork 2 — equal total scalar-probe budget

The next attack fixes the actual resource.

Dimensions:

```text
16, 32, 64, 128, 256
```

Budgets:

```text
64, 128, 256, 512 scalar relevance evaluations
```

Every scalar center or +/- directional evaluation costs one unit. Methods stop instead of exceeding the budget.

This gives the 8-direction block methods many cheap fast cycles while a full D-coordinate gradient may get only a few—or no—updates.

The relevance signal remains noiseless in this fork so the **budget/horizon** issue is isolated cleanly. If compressed probes survive, a scalar-measurement-noise attacker follows immediately.

Implementation: `experiments/fork_highdim_equal_probe_budget.py`.

## Fork 2 result — the 10-cycle collapse was partly a budget artifact

Holding total scalar evaluations fixed changes the story sharply.

At 128 dimensions with a 512-evaluation budget:

```text
dense hidden correction
full coordinate        final 0.465   success  0.0%   1 complete update
coordinate block8      final 0.181   success 87.5%
Hadamard block8        final 0.167   success 95.8%
random orthogonal8     final 0.314   success  0.0%
SPSA two-probe         final 0.932   success  0.0%
no probing             final 0.605   success  0.0%
```

For a sparse four-coordinate hidden correction at the same dimension/budget:

```text
full coordinate        final 0.467   success  0.0%
coordinate block8      final 0.301   success  0.0%
Hadamard block8        final 0.189   success 75.0%
no probing             final 0.607   success  0.0%
```

So the fixed-cycle failure was not an O(D) lower bound. A sequence of small structured measurements can use the same total scalar budget much more effectively than insisting on a complete coordinate gradient before every update.

But the result is not dimension-free. At 256 dimensions / 512 evaluations, none of the methods reaches the 0.18 success radius:

```text
dense:
coordinate block8      final 0.288
Hadamard block8        final 0.296
no probing             final 0.604

sparse4:
coordinate block8      final 0.763
Hadamard block8        final 0.266
no probing             final 0.605
```

The full-coordinate method cannot make even one 256-D update because a complete center + +/- coordinate batch costs 513 scalar measurements.

That sparse result is particularly useful: Hadamard mixing is not merely another random compressed direction set. It can expose a correction distributed over unknown coordinates much better than sequential coordinate coverage under the same short horizon.

Receipt: `results/fork_highdim_equal_probe_budget_summary.json`.

## Fork 3 — noisy scalar consequence

The equal-budget experiment still gave the sampler an unrealistically clean scalar consequence. In high dimensions each individual +/- perturbation can change relevance by only a small amount, so exact floating-point comparison can hide the real bottleneck.

The next attack therefore keeps:

```text
same hidden correction
same 512 scalar-evaluation budget
same structured partial probes
same fast update rule
```

and changes only:

```text
measured relevance
    = true relevance
    + independent Gaussian noise
```

Noise standard deviations:

```text
0, 0.005, 0.01, 0.02, 0.04
```

The machine no longer receives privileged true error when deciding whether it is already calibrated. The attackers are:

- full coordinates;
- progressive coordinate blocks;
- progressive Hadamard blocks;
- Hadamard with every +/- measurement repeated twice;
- Hadamard with adaptive repetition until the sign of the pair difference is reasonably resolved;
- random orthogonal blocks;
- random-sign blocks;
- SPSA-style two-probe updates;
- no directional probing.

Implementation: `experiments/fork_highdim_measurement_noise.py`.

### Fork 3 result — there is a dimension x consequence-SNR boundary

The clean 128-D result survives tiny measurement noise, but not indefinitely.

Dense correction, 128-D, 512 scalar measurements:

```text
noise sigma      Hadamard     repeat x2     adaptive repeat    no probing
0.000              0.155         0.198            0.155            0.605
0.005              0.183         0.202            0.168            0.605
0.010              0.364         0.213            0.212            0.605
0.020              0.345         0.253            0.241            0.605
0.040              0.395         0.372            0.440            0.605
```

At sigma 0.005, plain Hadamard still reaches the 0.18 success radius in 84.4% of the trials. At sigma 0.01 the single-measurement estimate becomes unreliable; spending measurements on repeated evidence is suddenly worth more than spending them on more fast updates.

The sparse hidden-correction case preserves the Hadamard advantage even more clearly at 128-D:

```text
noise sigma      coordinate     Hadamard     adaptive repeat
0.000               0.308          0.184           0.184
0.005               0.312          0.161           0.147
0.010               0.331          0.193           0.156
0.020               0.405          0.355           0.231
0.040               0.585          0.383           0.425
```

At 256-D the boundary is much harsher. For dense correction, plain Hadamard degrades from 0.292 with no measurement noise to 0.355 at sigma 0.01, 0.511 at 0.02, and 0.777 at 0.04. The no-probe baseline is about 0.605. At the highest noise level, blindly trusting the partial probes is therefore worse than doing nothing. Adaptive repetition retreats almost exactly to baseline (0.604) rather than rescuing useful calibration.

For sparse4 at 256-D, Hadamard remains useful longer:

```text
noise sigma      coordinate     Hadamard     adaptive repeat    no probing
0.000               0.766          0.253           0.253            0.605
0.005               0.808          0.270           0.374            0.605
0.010               0.825          0.319           0.451            0.605
0.020               0.847          0.474           0.526            0.605
0.040               0.877          0.772           0.601            0.605
```

So there is now a cleaner boundary than "high dimension is hard":

> **Useful fast internal probing depends on the ratio between the consequence change induced by a probe and the noise on the consequence signal.**

The structured basis can postpone the dimensional collapse, especially for sparse hidden corrections, but it cannot abolish that information problem.

A second useful result falls out of the repetition attack:

> **More observations are not automatically better.**

At low noise, repeating every probe wastes the finite budget and hurts. At intermediate noise, repetition helps dramatically. At very high dimension/noise, even repetition consumes too much budget to recover the missing directional information.

Receipt: `results/fork_highdim_measurement_noise_summary.json`.
CI source run: `33041717524`.

## Next fork — let the sweep width become a control variable

This is where the two neuroscience papers become useful again without pretending that our toy is a hippocampus.

The Ji model uses firing-rate adaptation to make an attractor bump intrinsically mobile, while theta controls the rhythm of that motion. The newer Vollan data show that biological sweeps do not have one fixed geometry: direction, width and frequency can all be retuned according to current behavioral demand.

Our engineering analogue now has a concrete reason to vary **width**.

The current probe radius is fixed at 0.28. But the noisy-consequence result says the sampler faces a tradeoff:

```text
small probe radius
    -> local / precise
    -> weak scalar difference
    -> bad consequence SNR

large probe radius
    -> stronger scalar difference
    -> easier sign estimate
    -> less local / potentially biased correction
```

So the next question is:

> **Can a fast sampler actively widen its probes when consequence SNR is poor and narrow them again near a confident solution, outperforming every fixed probe radius under the same scalar-measurement budget?**

That is a much better reason for an attention-like variable than simply copying theta sweeps because brains have them.

Planned attackers:

```text
fixed radius: 0.10 / 0.20 / 0.28 / 0.40 / 0.60
multi-scale scheduled radius
noise-aware adaptive radius
adaptive radius + adaptive repetition
```

The kill condition is straightforward: if one fixed radius or an ordinary finite-difference schedule matches the adaptive controller, the extra mechanism has not earned its place.
