# Structured sampling fork

Starting point: `control-law-fork` conclusion at `7f086f0`.

Question:

> Under equal wall time, equal probes, and equal per-cycle path length, is alternating left/right coverage special?

First bakeoff compares path-matched deterministic and quasi-random trajectories around the same noisy directional cue:

- one-sided alternating triangle;
- one-sided alternating sine;
- bilateral triangle;
- bilateral sine;
- two-harmonic oscillator;
- golden-phase sine;
- van-der-Corput low-discrepancy waypoint order;
- van-der-Corput points sorted into a low-travel coverage path;
- center-out interleaving;
- bounded smooth random walk;
- axis-only;
- unmatched IID random upper bound.

Worlds:

1. reliable directional cue;
2. changing cue reliability;
3. systematic cue bias;
4. cue loss and return.

All path-matched schedules are rescaled **every cycle** to the same total variation budget, `2 * uncertainty_radius`. IID random is intentionally not matched and must be interpreted together with its much larger travel.

This branch is exploratory; canonical `main` is untouched.
