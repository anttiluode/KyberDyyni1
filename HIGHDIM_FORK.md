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
