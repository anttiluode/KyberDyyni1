# Paper map

These papers are biological inspiration and falsification constraints, not implementation specifications.

## Ji, Chu, Wu & Burgess — Current Biology (2025)

**A systems model of alternating theta sweeps via firing rate adaptation**

DOI: https://doi.org/10.1016/j.cub.2024.08.059

Architecture described in the paper:

```text
theta-modulated internal-direction ring attractor
                    |
                    v
       conjunctive grid x direction cells
                    |
          shifted phase input
                    |
                    v
          grid-cell 2-D attractor
```

Key ingredients relevant here:

- continuous attractor dynamics;
- firing-rate adaptation;
- medial-septal theta modulation;
- upstream internal-direction sweeps driving downstream location sweeps;
- continuously moving bumps rather than a sequence of independent forward passes.

KyberDyyni1 currently uses a much simpler engineered alternator. Reproducing the sweep from adaptation is an explicit future gate.

## Vollan, Schellenberger, Moser & Moser — bioRxiv (2026)

**Attention-like regulation of theta sweeps in the brain's spatial navigation circuit**

DOI: https://doi.org/10.64898/2026.01.27.702083

Relevant reported observations:

- sweep direction, width and frequency are rapidly and dynamically modulated;
- pursuit produces narrower, faster sampling;
- internal direction/sweeps can orient toward a target before overt reorientation;
- sweeps reverse with backward locomotion;
- internally generated modulation persists in REM;
- stable upstream head-direction signals can coexist with a more elastic parasubicular/MEC internal-direction system.

Engineering abstraction:

```text
stable anchor/reference
        +
elastic internal sampler
        +
slow learning
```

## What is inference, not a paper claim

The following are KyberDyyni hypotheses:

- treating theta phase as an explicit local credit/address coordinate;
- mapping the fast sweep system onto a generic AI "proposal" or internal-search mechanism;
- mapping slower structural change onto a cortex-like consolidator;
- using the same scanner for online and offline artificial computation.

Those ideas must be tested separately.
