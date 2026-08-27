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
