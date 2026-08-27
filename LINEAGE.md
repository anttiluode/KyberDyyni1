# Lineage note — the two-ChatGPT incident

On 2026-08-27 this repository went through an unusual response-comparison event: two ChatGPT variants were simultaneously reasoning about the same project, and both had access to the GitHub repository during overlapping work.

This document exists so that later archaeology does not turn that accident into fake scientific provenance.

## What actually exists in Git

There is only one branch: `main`.

The repository history is linear. The selected lineage is the line currently ending at the post-Gate-4 commits. There is no hidden sibling branch to merge and no second committed Gate-3 implementation to recover.

The sibling model saw the already-committed Gates 0–2 and CI, then independently built/tested a Gate-3-like circular attractor locally. Its attempt to create the new Gate-3 source file on GitHub failed. It therefore offered an applicator rather than claiming the code had landed.

The current lineage subsequently committed its own Gate 3 and Gate 4, with saved receipts and CI reproduction.

## Where both lines agreed

Both lines converged on the same important boundaries:

- Gate 2 should make phase earn a real job by distinguishing otherwise-identical within-sweep events under delayed consequence.
- An explicit symbolic slot address is a decisive attacker and also solves Gate 2, so the earned claim is **phase as a viable address**, not phase superiority.
- The engineered left/right parity scanner is too cheap a cheat to support a biological-dynamics claim.
- A Ji-inspired recurrent ring with slow adaptation and theta modulation can generate alternating bump motion without a left/right instruction.
- The next useful question is not more biological decoration. It is whether a fast internally generated sweep helps a slow-learning machine do useful work.

That convergence is worth preserving.

## Where the quantitative Gate-3 stories differ

The sibling response reported a local six-seed result roughly as:

```text
adaptation + theta:
  mean sweep displacement       27.24° ± 0.33°
  cycle-to-cycle alternation    1.000

no adaptation:
  mean displacement              0.0035°

no theta:
  mean displacement              0.0355°
```

Its exact source and measurement code are not in Git, so these numbers are **historical, noncanonical evidence**.

The committed Gate 3 uses ten seeds and saves:

```text
full mechanism:
  mean theta-cycle peak          0.5931 rad ~= 34.0°
  side alternation               1.000

no adaptation:
  mean theta-cycle peak          0.0025 rad

no theta:
  mean theta-cycle peak          0.0135 rad

no recurrence:
  mean theta-cycle peak          0.0252 rad
```

These should not be treated as conflicting replications. The committed receipt's amplitude is specifically the maximum absolute bump displacement inside each theta cycle, averaged over cycles. The sibling called its quantity "mean sweep displacement" and its implementation/metric is unavailable.

The safe conclusion shared by both is qualitative: **large organized alternating motion appears with the full adaptation+theta mechanism and collapses under the principal ablations.**

Only files under `results/` count as canonical quantitative receipts.

## What the sibling contributed to the surviving roadmap

The most useful idea from the sibling line was its proposed next architecture:

```text
stable anchor
      |
adaptation-generated sweep
      |
candidate events at phase positions
      |
immediate relevance
      |
fast retarget / focus / frequency change
      |
action
      |
later consequence
      |
phase/context-local trace
      |
slow consolidation
```

The current line's Gate 4 tested only the middle slice: **can the continuous sweep perform useful zeroth-order search before slow learning?**

It can, but engineered sweep and random dither beat it on raw acquisition speed. That result should stay; it is an important attacker win.

The sibling's proposal therefore becomes Gate 5 rather than replacing Gate 4:

> **First make relevance alter the fast sampling policy with slow weights frozen. Then let delayed success consolidate a bounded context prior so the next encounter begins in a better place.**

That is the cleanest concatenation of the two lines.

## Provenance rule going forward

When two implementations or model variants disagree:

1. keep the committed/reproducible receipt as canonical;
2. preserve uncommitted sibling results as provenance, not data;
3. steal good experimental attacks and hypotheses freely;
4. never tune away an attacker win merely because another line told a prettier story.

The response-comparison event was funny. The repo should remain less confused than the event that produced it.
