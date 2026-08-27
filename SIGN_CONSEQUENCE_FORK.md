# Sign-consequence calibration fork

Gate 11 left one clean identifiability boundary.

For symmetric Gaussian AR sources, blind temporal decomposition can recover:

- the low-rank source axes;
- which component is which across independent views;

but it cannot determine the orientation of each axis because the process is exactly invariant under a sign flip.

For rank 4 the remaining uncertainty is only:

```text
4 global sign bits
= 16 possible orientation maps
```

That is much smaller than the original 16-D coordinate relation.

## Question

> **Can local scalar consequence resolve those residual sign bits, then retain the result as one reusable cross-view map?**

The experiment first performs the same unlabeled Gaussian-AR temporal alignment as Gate 11. It then generates unrelated calibration contexts.

For each candidate sign map the machine receives only:

```text
scalar relevance = exp(-error^2 / (2 sigma^2)) + noise
```

It never receives the hidden B-side correction vector except in the explicit paired-vector oracle.

Once sign bits are learned, the same global orientation map is tested on 256 unseen contexts.

## Attackers

- fixed random signs;
- simple hill climbing over one sign flip at a time;
- independent +/- evidence for each sign bit;
- the same bitwise evidence with two repeated scalar measurements;
- exhaustive evaluation of all 16 sign patterns;
- shuffled bitwise consequence;
- one fully paired correction vector as an oracle;
- exact sign oracle as a diagnostic.

Consequence noise:

```text
0.00, 0.01, 0.02, 0.04
```

Calibration horizons:

```text
0, 1, 2, 4, 8, 16, 32 contexts
```

## What would count

The useful result is not a fancy sign learner.

If an ordinary eight-scalar bitwise comparison solves the residual ambiguity, accept it.

The architectural claim would instead be:

```text
hard cross-basis relation
       |
temporal source structure
       v
axes + permutation
       |
local scalar consequence
       v
few global sign bits
       |
slow retention
       v
cheap future transfer
```

## Kill conditions

- random or shuffled consequence performs similarly to correct consequence;
- sign evidence does not generalize to unseen contexts;
- a new context requires a fresh sign search every time;
- the temporal decomposition error is too large for even the oracle sign map to transfer;
- ordinary exhaustive / bitwise search solves the problem, but we nevertheless pretend the update rule is novel.
