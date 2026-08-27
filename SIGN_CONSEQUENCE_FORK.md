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


## Result — the residual sign ambiguity is cheap to finish with scalar consequence

The bitwise attacker wins cleanly.

After the Gaussian-AR temporal decomposition, the learned cross-view geometry is already accurate enough that the exact sign oracle transfers unseen contexts with:

```text
error   0.0376
success 100%
```

So the remaining problem really is almost entirely the four sign bits.

### No consequence noise

```text
method                    contexts   scalar evals   sign accuracy   transfer success

random signs                  --           0           52.1%             15.8%
hill climb                     1           5           96.4%             88.3%
bitwise                        1           8           95.8%             86.6%
bitwise                        2          16          100.0%            100.0%
exhaustive 16 patterns         2          32          100.0%            100.0%
shuffled consequence          32         256           51.6%             13.8%
```

Two unrelated calibration contexts and **16 scalar measurements total** are enough for ordinary bitwise evidence to recover the complete reusable orientation map.

### Noisy consequence

The same mechanism degrades gracefully.

```text
consequence noise      contexts to reach oracle floor      scalar evals

0.00                               2                         16
0.01                               4                         32
0.02                               8                         64
0.04                              16                        128
```

Selected receipts:

```text
sigma 0.01
K=2   bitwise success 94.8%   sign accuracy 98.4%
K=4                   100%                   100%

sigma 0.02
K=4                   96.6%                  99.0%
K=8                    100%                  100%

sigma 0.04
K=4                   96.6%                  99.0%
K=8                   98.3%                  99.5%
K=16                   100%                  100%
```

Repeated measurements of every +/- candidate do not buy enough to justify their doubled cost here. Ordinary accumulation across **different contexts** is better because different contexts naturally excite different components strongly.

That is a useful connection to the architecture: experience diversity itself supplies repeated evidence.

### Shuffled consequence control

The causal control fails exactly as it should.

Even after 32 contexts / 256 scalar evaluations:

```text
sign accuracy ~49--52%
transfer success ~11--16%
transfer error ~0.74--0.77
```

Correct local consequence is therefore doing the work; the result is not an artifact of the temporal basis estimator drifting toward the right signs on its own.

### One paired vector is strong, but not perfect

A single fully observed paired correction vector resolves about 97.9% of the sign bits on average and gives about 93.3% transfer success.

That sounds surprising until the reason is visible: one random context can have a coefficient very close to zero on one component, so even a complete vector pair may not strongly expose every sign bit.

Several scalar-consequence contexts can actually do better because they excite different components.

## What Gate 12 earns

Again the learning rule is boring.

The useful decomposition is not:

> "we invented a new sign optimizer."

It is:

```text
arbitrary cross-view basis relation
             |
      temporal structure
             v
 axes + component identity
             |
   exact source symmetry leaves
       R global sign bits
             |
      scalar local consequence
             v
       reusable polarity map
```

For the rank-4 toy, temporal structure plus 16--128 ordinary scalar measurements—depending on consequence noise—turns a blind unusable map into oracle-level transfer.

This closes the Gaussian sign ambiguity without supplying context-pair labels or synchronized trajectories.

The remaining attacker is now obvious:

> **Does this still scale when the transferable family has more than four components?**

Exhaustive search becomes 2^R, while bitwise evidence costs only O(R) measurements per calibration context. The next gate should attack rank, sparse excitation, and weakly observed components rather than add another mechanism.
