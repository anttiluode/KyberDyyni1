# Structured fork conclusion

This fork began by attacking the claim that left/right alternation itself was computationally special.

It found a narrower reason that survives the attackers.

## 1. Alternation is not special when trajectories are artificially closed

Under identical per-cycle path budgets, bilateral and sorted low-discrepancy sweeps beat a one-side-per-cycle alternator if every cycle has to return to center.

So "left one cycle, right the next" does not win merely because it alternates.

## 2. Persistence across cycle boundaries changes the geometry

Once the sampler's endpoint is allowed to remain state:

```text
left  ------------------------------> right
right ------------------------------> left
```

the shuttle covers the complete 1-D uncertainty interval without paying a return-to-center cost.

Across reliable cues, changing reliability, systematic bias and loss/reacquisition, it approaches IID random coverage while using roughly 1/30 of the path travel.

A matched-travel smooth random walk performs far worse.

The surviving operation is therefore:

> **persistent, structured boundary-to-boundary coverage**

not random smoothness and not a particular neural implementation.

## 3. Delayed credit exposes the coordinate-frame problem

A reversing trajectory means raw theta phase changes spatial meaning every cycle.

Delayed local learning succeeds only after phase is oriented by current sweep direction:

```text
oriented phase      95.5%
raw phase           36.9%
no/shuffled phase   ~12.5%
explicit oriented   100%
```

Therefore the natural artificial address is not "phase" in isolation.

It is:

```text
event address = trajectory phase × trajectory orientation
```

or any equivalent local coordinate.

## Current architecture

```text
                STABLE REFERENCE
                       |
               fast control vector
                       |
                       v
             CONTINUOUS SHUTTLE
             <---------------->
                       |
               candidate events
                       |
           oriented local coordinate
                       |
                  time passes
                       |
               scalar consequence
                       |
                       v
               SLOW PREFERENCE
```

This is now an engineering architecture assembled from independently attacked pieces, not a claim that the artificial system is a hippocampus.

The obvious next attack is dimensionality.

In one dimension, boundary-to-boundary traversal is almost geometrically privileged. In 2-D or a higher-dimensional latent control surface, there is no single "left boundary -> right boundary" path that covers the uncertainty set.

The next fork should therefore ask:

> **What replaces alternating sweeps when the uncertainty manifold is 2-D?**

Raster scan, boustrophedon paths, spirals, Hilbert/Peano-like curves, low-discrepancy points connected by short tours, coupled oscillators and learned trajectories should all be attacked under equal path and delayed-address budgets.

If the whole discovery collapses outside 1-D, that is an important boundary.
