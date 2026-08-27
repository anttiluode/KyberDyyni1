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

KyberDyyni1 now contains both versions. `ThetaScanner` is the deliberately engineered alternator; `AdaptationRingScanner` ports the paper's head-direction ring mechanism into NumPy using the authors' published equations and their simple-demo parameterization. Gate 3 reproduces large cycle-to-cycle alternating sweeps only when recurrence, adaptation, and theta modulation act together.

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


## Reproduction note for Gate 3

The Ji et al. paper states that the HD and grid networks use fast rate dynamics plus slower firing-rate adaptation, with adaptation providing intrinsic bump mobility and theta constraining the sweep rhythm. Its STAR Methods give the rate/adaptation equations and specify a 10-ms rate time constant and 100-ms adaptation time constant for the simulations.

The authors' public repository `ZilongJi/GridCellThetaSweeps` provides `Network_models.py` and `HD_phase_shift.py`. Gate 3 follows the HD module there, including its circular Gaussian recurrent kernel, rectified activation, divisive global inhibition, adaptation state, theta-modulated Gaussian anchor input, and the demonstration parameters `tau=10`, `tau_v=100`, `mbar=12`, `a=0.4`, `A=3`, `J0=4`, 100 cells, 1-ms dt, 100-ms theta period, and theta modulation strength 0.4.

KyberDyyni does not copy their BrainPy/JAX implementation or full grid-cell network; the repo uses a small NumPy port so the mechanism can be attacked in isolation.
