# KyberDyyni1 — fast sweeps, phase addresses, slow structure

A new architecture branch from [T-800NNP](https://github.com/anttiluode/T-800NNP).

T-800NNP established a small set of useful primitives:

- a continuously running receiver can carry temporal history;
- present-time-identical events can be routed differently because receiver history differs;
- events need not reset the receiver;
- delayed local consequence can change bounded structure without BPTT/autograd;
- the stream can run without artificial episode resets.

KyberDyyni1 asks the next question:

> **What changes if the fast internal dynamics are not merely memory of the past, but an active phase-organized sampler that sweeps through nearby possible states, while slower structure consolidates what repeatedly matters?**

The biological inspiration is the entorhinal–hippocampal theta-sweep literature, especially:

- Ji, Chu, Wu & Burgess (Current Biology, 2025), *A systems model of alternating theta sweeps via firing rate adaptation*.
- Vollan, Schellenberger, Moser & Moser (bioRxiv, 2026), *Attention-like regulation of theta sweeps in the brain's spatial navigation circuit*.

This repo is **not** a hippocampus simulator and does not claim that entorhinal cortex literally stores phase tags in the form used here. The papers motivate architectural tests; the tests must earn the engineering claims.

A strange provenance note: during ChatGPT response comparison on 2026-08-27, two model variants briefly reasoned over the same repository. The selected line is the one now on `main`; useful ideas from the sibling line are preserved explicitly rather than silently blended. See [LINEAGE.md](LINEAGE.md).

## Motivation

The motivating neuropsychological analogy is the classic observation that severe hippocampal damage can leave much ongoing cognition intact while profoundly impairing formation of new long-term declarative memories.

We are not equating a normal artificial neural network with "cortex minus hippocampus." The useful engineering question is narrower:

> Can one machine separate **fast elastic state computation** from **slow structural learning**, rather than asking one weight system to do both?

## Architecture hypothesis

```text
external / cortical anchor A(t)
            |
            v
      FAST STRUCTURED SAMPLER H(t)
      - continuous trajectory state
      - low-travel coverage
      - fast retargeting / uncertainty control
      - biological theta is one possible generator
            |
            v
      TRAJECTORY ADDRESS E(t)
      - where in the unfolding path?
      - in which path orientation / coordinate frame?
      - local address for later credit
            |
      +-----+------+
      |            |
      v            v
candidate      eligibility /
outputs        local traces
      |            |
      +-----+------+
            |
        consequence
            |
            v
      SLOW STRUCTURE M
      - bounded
      - local update
      - no BPTT/autograd
      - biases future scans
```

Three timescales are intentionally separated:

1. **Fast state** — can change immediately without learning.
2. **Trajectory-relative address** — binds an internal sample to its location in an ongoing trajectory. Gate 2 showed phase can do this in a fixed sweep; Gate 6 shows a reversing sweep requires phase to be oriented by the current trajectory direction.
3. **Slow structure** — changes only after repeated/delayed consequence.

## What the papers actually motivate

### Ji et al. 2025

Their model combines:

- a theta-modulated internal-direction ring attractor;
- an intermediate conjunctive grid × direction layer;
- a grid-cell continuous attractor;
- firing-rate adaptation;
- medial-septal theta modulation.

Adaptation destabilizes an activity bump and gives it intrinsic mobility. Theta controls the rhythm. The upstream internal-direction sweep drives a downstream location sweep through a shifted phase input.

KyberDyyni1 steals the **functional decomposition**, not the biological implementation.

`ThetaScanner` remains the deliberately simple engineered baseline. Gate 3 now adds `AdaptationRingScanner`, a NumPy mechanism port in which the alternating sweep emerges from recurrent attraction, slow adaptation and theta modulation rather than a parity rule.

### Vollan et al. 2026

The newer experiments motivate treating the sweep as a controllable internal sampler. The reported sweep system can rapidly change:

- **direction**,
- **width**,
- **frequency**,

and can retarget during pursuit, orient before overt movement, reverse during backward locomotion, and continue to express internally generated modulation during REM.

That suggests a useful artificial primitive:

```text
broad + slower scan     -> explore
narrow + faster scan    -> inspect / focus
axis redirected         -> attend elsewhere
```

without requiring immediate slow-weight changes.

## Gate 0 — theta scanner without learning

`experiments/gate0_theta_scanner.py`

The scanner is allowed to retarget purely through fast state.

Typical deterministic receipt:

```text
exploratory:
  axis       0.00
  width      1.00
  frequency  0.08
  ~31 cycles / 400 steps

focused right:
  axis       1.00
  width      0.35
  frequency  0.14
  ~55 cycles / 400 steps

released:
  axis       0.00
  width      1.00
  frequency  0.08

slow weight changes: 0
```

This gate earns only:

> **Internal sampling policy can change quickly at the state level without changing slow structure.**

It does not earn biological theta dynamics.

## Gate 1 — fast response, then slow consolidation

`experiments/gate1_fast_slow_consolidation.py`

Two contexts require opposite scan axes.

First, a fast control signal tells the scanner where to look while **slow learning is disabled**. It can solve the immediate retargeting problem even though the slow context weights have not learned the mapping.

Then the fast cue is removed before consolidation: performance returns to chance.

Finally, repeated delayed local targets slowly teach the context -> axis prior. The fast cue is removed again and the slow structure can now reproduce the useful bias.

10-seed receipt:

```text
fast control before slow learning       0.8930
no cue before consolidation             0.5023
no cue after slow consolidation         0.8912
```

This earns a small but useful separation:

> **Fast state can solve an immediate problem before slow weights know it; repeated delayed local teaching can later consolidate the mapping.**

The first phase attack failed here: shuffling a short delayed context-address queue hardly hurt because neighboring samples usually belonged to the same context. That failure forced Gate 2 to remove temporal-neighbor ambiguity.

## Gate 2 — phase is an address, not decoration

`experiments/gate2_phase_address.py`

One continuous theta clock generates **eight candidate internal events per cycle**. Within a cycle the local event content and context are deliberately held identical; only their theta phase distinguishes them.

Four contexts each prefer a different candidate slot. The machine chooses one event, but reward arrives **four theta cycles later**. The retained local credit record must therefore identify which within-sweep event was chosen.

10 seeds:

```text
phase address                    1.0000 ± 0.0000
no phase address                 0.1219 ± 0.0128
shuffled phase address           0.1192 ± 0.0061
8-way chance                     0.1250
explicit slot-address attacker   1.0000 ± 0.0000
```

This finally earns a narrow phase claim:

> **When otherwise identical candidate events coexist inside one internally generated sweep, theta phase can serve as a stable local address that lets delayed scalar consequence discriminate among them.**

But the explicit slot attacker also solves the task perfectly. Therefore Gate 2 does **not** show that oscillatory phase is superior to ordinary addressing.

It shows something more specific and useful:

> **phase has earned a computational job: temporal addressability.**

The slow update is reward-modulated local competition. A candidate's local credit vector is retained; several theta cycles later a scalar consequence changes bounded weights. No intervening state sequence is replayed or differentiated through.

## Phase hypothesis: current status

`PhaseBridge` and `PhaseAddressedSelector` now implement two related ideas:

- phase as a coordinate describing **where an event lies inside a fast internal trajectory**;
- phase × context conjunctions as a local address for slow plasticity.

Gate 2 supports the first use in a deliberately isolated task.

Still unearned:

- phase outperforming an explicit counter/slot address;
- phase solving arbitrary long-delay credit;
- phase being necessary once candidate contents themselves are distinctive;
- any claim that biological entorhinal cortex implements this exact code.

## No backpropagation

There is no PyTorch, JAX, TensorFlow, autograd, gradient tape, reverse graph traversal, or BPTT.

The slow learners use local retained addresses / eligibility-like records plus delayed consequence:

```text
fast local event
      |
phase/context address
      |
retained local credit
      |
   time passes
      |
scalar consequence
      |
bounded local update
```

Gate 2 uses a reward-modulated local competition rule. That is still a learning rule with a teaching/reward signal; it is not a solution to arbitrary deep credit assignment.

## Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_theta_scanner.py
python experiments/gate1_fast_slow_consolidation.py
python experiments/gate2_phase_address.py
python experiments/gate3_adaptation_attractor.py
python experiments/gate4_internal_search.py
python experiments/gate5_fast_slow_memory.py
python experiments/fork_cross_cycle_sequences.py
python experiments/fork_phase_direction_address.py
python run_all.py
python -m unittest discover -s tests
```

Results are saved in `results/`. Saved JSON receipts are the canonical quantitative record. Historical/sibling-model results that were never committed are documented separately in [LINEAGE.md](LINEAGE.md), not mixed into `results/`.

## Gate 3 — adaptation generates the sweep

`experiments/gate3_adaptation_attractor.py`

The explicit left/right parity rule is removed. `AdaptationRingScanner` is a pure-NumPy mechanism port of the theta-modulated head-direction ring used in Ji et al.'s published model/code:

```text
Gaussian recurrent ring attraction
          +
stable external anchor
          +
slow firing-rate adaptation
          +
sinusoidal theta modulation
          ↓
moving population bump
```

Default parameters mirror the authors' simple HD demonstration: 100 cells, 1-ms integration step, 10-ms rate time constant, 100-ms adaptation time constant, 100-ms theta period, adaptation strength `mbar=12`, recurrent gain `J0=4`, and theta modulation 0.4.

10 seeds after a 1-s burn-in:

```text
                                   mean theta-cycle peak     side alternation
full mechanism                      0.5931 rad (~34.0°)          1.000
no firing-rate adaptation           0.0025 rad                   0.457
no theta modulation                 0.0135 rad                   0.757
no recurrent attraction             0.0252 rad                   0.541
```

The canonical Gate-3 amplitude metric is **the mean, across theta cycles, of the largest absolute bump displacement reached inside each cycle**. A sibling ChatGPT prototype produced a separate local Gate-3 result around 27° using an uncommitted implementation/measurement. That is useful as an independent qualitative cross-check, but it is not numerically comparable to the saved receipt above. See [LINEAGE.md](LINEAGE.md).

There is no `left`, `right`, cycle parity, or alternating-sign instruction in this scanner. The full system spontaneously settles into large bounded sweeps that reverse side every theta cycle.

This earns:

> **A fast alternating sampler can be generated by continuous population dynamics rather than imposed as an external sequence. Adaptation creates mobility, recurrence supports a coherent bump, and theta organizes that mobility into a stable rhythm.**

It does **not** show that this population is more useful than the tiny engineered `ThetaScanner`. The engineered scanner remains dramatically simpler and alternates perfectly by construction. Gate 4 must make the dynamical scanner perform useful search before the added mechanism has earned its engineering cost.

## Gate 4 — useful internal search, with an attacker win

`experiments/gate4_internal_search.py`

The fast sampler now performs an actual operation rather than merely producing a sweep. A hidden target lives on a circular latent manifold and jumps eight times per seed. The system is never given target direction or a derivative. It receives only the scalar value of whichever internal point it probes.

All active searchers use the same fast zeroth-order steering rule:

```text
anchor += eta * (sample_value - running_baseline) * (probe - anchor)
```

No slow weights change.

10 seeds:

```text
                              acquisition     success     tracking error   probe travel / step
adaptation attractor          554 steps        95.0%       0.443 rad        0.0116 rad
engineered theta sweep        427 steps        98.8%       0.330 rad        0.0117 rad
random dither                 313 steps       100.0%       0.240 rad        0.4000 rad
static / no sweep            1170 steps         2.5%       1.757 rad        0
```

So the adaptation-driven scanner **does useful derivative-free internal search**, but it does not win the simple benchmark. The trivial engineered sweep is faster, and random dither is faster still.

The interesting tradeoff is continuity: the attractor and engineered sweeps move only about 0.012 rad per step, while random dithering moves about 0.40 rad per step. Random search is roughly 34x more mobile internally in exchange for its faster acquisition.

This earns:

> **A continuously generated fast sweep can be used as an online search process before slow learning changes anything.**

It does **not** earn:

> the biological sweep mechanism is a better generic optimizer.

If internal trajectory continuity, transition cost, or state-dependent computation does not matter in later tasks, the random/engineered attackers should kill the extra attractor machinery.

## Gate 5 — fast state becomes slow memory

`experiments/gate5_fast_slow_memory.py`

Gate 5 finally asks the architectural question directly:

> **Can fast continuously evolving state do useful temporary computation, and can delayed local consequence crystallize successful fast states into slow structure without BPTT?**

Six recurring contexts each hide a noisy target location on a circular latent manifold. The scanner receives no target direction or derivative. A self-generated probe receives only scalar relevance. The continuous attractor population is never reset between probe steps.

### 5A — useful computation exists before learning

Slow structure is frozen at exactly zero.

```text
                                      first acquisition   success   mean sample value
fast directional state                    92.6 steps      100.0%       0.921
no fast relevance                        700.0 steps        0.0%       0.503
naive axis + focus modulation             334.1 steps       64.6%       0.884
```

The directional fast state therefore earns a strong result: useful target-seeking behavior appears during the encounter while **slow capacity remains 0**.

The naive Vollan-inspired focus control did *not* help this task. It sped theta and reduced adaptation to narrow the sweep, cutting probe travel from about 0.0146 to 0.0049 rad/step, but acquisition became much slower and less reliable. Biological modulation is therefore not automatically an engineering optimization.

### 5B — delayed success changes the next encounter

After each encounter, the best self-generated probe is retained as a tiny local credit packet. Its scalar consequence is delivered only after **two later encounters**. No intervening state trajectory is replayed or differentiated through.

```text
                                      first acq   late acq   late starting error
bounded delayed slow prior              92.6       37.1          0.082 rad
slow structure frozen                   92.6       87.1          1.560 rad
credit context shuffled                 80.0      109.7          1.760 rad
explicit EMA table attacker             92.6       45.4          0.083 rad
engineered sweep + same slow prior      102.7       12.9          0.082 rad
random dither + same slow prior          94.1        1.6          0.090 rad
```

The bounded learner used about **5.73 L1 units** of slow capacity; the explicit EMA table used about **6.93** in this configuration. Their late starting errors are essentially the same. The small acquisition difference is not evidence that the bounded learner is generally superior; for one-hot recurring contexts the two slow learners are close cousins.

This earns the architectural claim:

> **Fast state can solve the immediate problem before slow learning occurs, and correctly addressed delayed local consequence can later crystallize that success into a prior that improves the next encounter.**

The shuffled-context ablation is decisive: updating slow structure is not enough; the delayed credit has to land on the right persistent context.

But the stronger biological-mechanism claim fails:

> **The adaptation-generated attractor is not required for this task.**

The engineered sweep is substantially faster after consolidation, and random dither is faster still. Random dither pays for that speed with about **0.400 rad/step** probe travel versus **0.0135 rad/step** for the adaptation scanner.

One important limitation: Gate 5's delayed credit packet is `(context, candidate probe, consequence)`. It does **not** require theta phase. Gate 2 remains the evidence that phase can address otherwise-indistinguishable within-sweep events; a later gate must integrate that address into this fast-to-slow memory task if phase is to remain central.

The full preregistered contract and interpretation are in [GATE5.md](GATE5.md), and the canonical numbers are in `results/gate5_fast_slow_memory.json`.

## Gate 6 — persistent structured trajectory + oriented address

The two experimental forks after Gate 5 were consolidated into one canonical result. Full histories remain in [FORK_CONCLUSION.md](FORK_CONCLUSION.md) and [STRUCTURED_CONCLUSION.md](STRUCTURED_CONCLUSION.md).

### 6A — continuity alone is not enough

Many artificial translations of Vollan-like focus were attacked. A separate fast directional/control signal made structured sampling useful as active sensing, but the Ji-like attractor was not required: a tiny deterministic alternator matched or beat it while using similarly low internal travel.

A smooth random walk at a comparable travel budget performed much worse.

So the useful property is not generic continuity:

> **structured low-travel coverage matters; generic smooth motion does not.**

### 6B — persistence across sweep boundaries matters

Under equal per-cycle path budgets, one-side alternation was *not* special if every cycle was forced back to center. Bilateral and sorted low-discrepancy sweeps were better.

Then the artificial reset was removed. The endpoint of one sweep became the start of the next:

```text
left  ------------------------------> right
right ------------------------------> left
```

Across reliable cues, changing reliability, systematic cue bias, and cue loss/return, this cross-cycle shuttle approached IID-random coverage while moving roughly thirty times less distance.

Selected receipts:

```text
RELIABLE CUE
IID random, unmatched          100.00% hit   ~0.327 rad/ms
cross-cycle shuttle             99.17%        ~0.011 rad/ms

MIXED RELIABILITY
IID random                       90.56%
cross-cycle shuttle              89.31%

SYSTEMATIC CUE BIAS
IID random                       99.58%
cross-cycle shuttle              95.69%

LOSS / RETURN
IID random                        0 ms reacquisition
cross-cycle shuttle               5.6 ms
smooth random walk              227.8 ms
```

This earns:

> **For a one-dimensional uncertainty interval, carrying trajectory state across sweep boundaries makes boundary-to-boundary traversal an extremely path-efficient coverage law.**

### 6C — phase must be expressed in the trajectory's coordinate frame

The reversing shuttle exposed a loophole in Gate 2. Raw phase no longer has a stable spatial meaning because the same phase occurs on opposite sides in opposite-direction cycles.

With reward delayed four cycles:

```text
oriented phase = phase bound to sweep direction    95.54%
raw theta phase                                     36.91%
explicit raw slot                                    50.14%
no phase                                             12.46%
shuffled oriented phase                              12.49%
explicit oriented slot                              100.00%
chance                                               12.50%
```

So Gate 2 is refined:

> **The useful address is a local coordinate in the currently unfolding trajectory. For a reversing 1-D shuttle, phase × sweep-direction is sufficient; raw phase is not.**

The symbolic oriented-slot attacker still wins perfectly, so phase is not claimed to be uniquely privileged.

Canonical evidence:

- `experiments/fork_cross_cycle_sequences.py`
- `results/fork_cross_cycle_sequences.json`
- `experiments/fork_phase_direction_address.py`
- `results/fork_phase_direction_address.json`
- [GATE6.md](GATE6.md)

## Gate 7 — 2-D structured probing, fast calibration, delayed slow calibration

The dimensionality attack is complete. Full details are in [GATE7.md](GATE7.md) and [TWO_D_CONCLUSION.md](TWO_D_CONCLUSION.md).

The 1-D boundary shuttle does **not** generalize as one privileged 2-D path. Under equal low travel, different curves win different uncertainty regimes:

- golden radial spokes are strong when the cue is approximately centered but noisy;
- spiral/Lissajous paths are stronger when the cue is systematically biased;
- generic smooth random walk remains poor;
- unmatched IID random remains the high-travel coverage upper bound.

A naive fast `miss -> switch geometry` controller failed. The binary miss signal detects trouble but does not say how the reference is wrong.

The useful 2-D operation came from keeping a structured local probe basis and letting relative sample relevance update an elastic correction vector.

Selected 12-seed means:

```text
                         fixed radial    + fast contrast

RELIABLE                    90.0%            92.8%
MIXED                       61.0%            82.5%
SYSTEMATIC BIAS             36.8%            85.1%

LOSS / RETURN
hit-cycle fraction          78.5%            86.3%
reacquisition               41.7 ms          30.6 ms
```

No slow weights change during that computation.

Repeated contexts can then crystallize the temporary correction after delayed consequence:

```text
                              FIRST        LATE

bounded delayed calibration
start error                   0.435        0.231
first-cycle hit               51.4%        82.6%

fast only / slow frozen
start error                   0.435        0.429
first-cycle hit               51.4%        38.2%

shuffled delayed context
start error                   0.438        0.455
first-cycle hit               50.0%        41.7%
```

The explicit EMA attacker is numerically identical to the current bounded slow learner because the structural budget does not activate.

Finally, a conventional finite-difference-like probe basis almost matches the golden radial path:

```text
                         radial     cardinal cross

RELIABLE                 92.8%          90.7%
MIXED                    82.5%          81.4%
SYSTEMATIC BIAS          85.1%          86.9%
LOSS reacquisition       30.6 ms        63.9 ms
```

So Gate 7 earns a simpler engineering statement:

> **A stable reference can be interrogated by a small structured local probe basis; relative relevance can alter fast calibration state immediately; repeated useful calibrations can later become delayed context memory.**

The digital architecture does **not** require golden-angle sampling, a Ji attractor, or a novel slow learner.

Canonical evidence:

- `experiments/fork_2d_fast_recenter.py`
- `results/fork_2d_fast_recenter.json`
- `experiments/fork_2d_fast_slow_calibration.py`
- `results/fork_2d_fast_slow_calibration.json`
- `experiments/fork_2d_probe_basis_attack.py`
- `results/fork_2d_probe_basis_attack.json`

## Gate 8 — high-dimensional scaling and consequence SNR

The dimensional attack is now complete. Full details are in [HIGHDIM_FORK.md](HIGHDIM_FORK.md).

The first 10-cycle experiment made the fixed-size structured probe bank look as if it simply collapsed with dimension. Equalizing the **actual scalar-relevance budget** changed that conclusion.

At 128 dimensions with 512 scalar evaluations:

```text
dense hidden correction

full coordinate        final 0.465    success  0.0%
coordinate block8      final 0.181    success 87.5%
Hadamard block8        final 0.167    success 95.8%
random orthogonal8     final 0.314    success  0.0%
SPSA two-probe         final 0.932    success  0.0%
no probing             final 0.605    success  0.0%
```

For a sparse four-coordinate hidden correction:

```text
coordinate block8      final 0.301    success  0.0%
Hadamard block8        final 0.189    success 75.0%
no probing             final 0.607
```

So the earlier fixed-cycle failure was partly a horizon artifact. A sequence of small structured measurements can use a finite scalar budget much more effectively than insisting on a complete coordinate gradient before every update.

But the mechanism is not dimension-free. At 256 dimensions / 512 evaluations, neither compressed method reaches the strict 0.18 success radius. Hadamard still reduces error substantially, especially for sparse hidden corrections, while a complete 256-D coordinate batch cannot even fit inside the budget.

The stronger boundary appeared when noise was added to the scalar consequence.

At 128-D dense:

```text
noise sigma      plain Hadamard    repeat x2    adaptive repeat
0.000                 0.155           0.198          0.155
0.005                 0.183           0.202          0.168
0.010                 0.364           0.213          0.212
0.020                 0.345           0.253          0.241
0.040                 0.395           0.372          0.440
```

At tiny noise, spending the budget on more directions is best. At intermediate noise, repeated evidence becomes worth more than additional movement. At sufficiently high dimension/noise, partial probes can become worse than doing nothing.

Gate 8 therefore earns a more precise statement:

> **Fast structured probing is limited by probe-induced consequence signal relative to consequence noise, not by latent dimension alone. Structured mixed probes can postpone the dimensional collapse, but cannot abolish the information limit.**

Canonical evidence:

- `experiments/fork_highdim_probe_scaling.py`
- `experiments/fork_highdim_equal_probe_budget.py`
- `experiments/fork_highdim_measurement_noise.py`
- `results/fork_highdim_probe_scaling.json`
- `results/fork_highdim_equal_probe_budget_summary.json`
- `results/fork_highdim_measurement_noise_summary.json`

### Probe-width subfork — negative result

The Vollan-style observation that biological sweep width can change dynamically motivated one further engineering test:

> If consequence SNR is poor, should the artificial sampler widen its probes, then narrow them again when evidence becomes clear?

That mechanism did **not** earn a place.

Under the harder 256-D conditions, a fixed probe radius around 0.40 beat the adaptive-width controllers across every tested noise level in both dense and sparse worlds. The literal low-SNR -> widen rule often made things worse by treating noisy evidence as a command to make still larger excursions.

So dynamic sweep width is frozen as a negative result in [PROBE_WIDTH_FORK.md](PROBE_WIDTH_FORK.md).

The biological paper motivated the question. The toy rejected the mechanism.

## Branch consolidation

The experimental branch history is now preserved inside the canonical tree rather than requiring branch archaeology.

See [BRANCHES.md](BRANCHES.md) for the frozen branch heads and the files brought into `main`.

The short version:

- control-law fork -> preserved;
- structured-sampling fork -> preserved;
- two-dimensional fork -> preserved;
- high-dimensional probe fork -> preserved;
- probe-width-control fork -> preserved;
- `main` remains the canonical narrative and implementation.

The old branches are still useful provenance, but no result now depends on remembering which branch contains it.

## Gate 9 — noisy fast search becomes slow amortized prior

Gate 9 reconnects the slow half of the architecture after Gate 8 deliberately froze it.

The world contains recurring contexts. Each context has a stable hidden correction of norm 0.60 plus a fresh nuisance correction of norm 0.08 on every encounter. Fast search receives only noisy scalar relevance through progressive Hadamard block probes. Slow memory never sees the true target or true correction.

The first attack exposed an important interaction failure.

Correctly addressed EMA/Kalman memories reduced late starting error from about 0.61 to about 0.17--0.24, while shuffled context credit made it worse. So slow memory was learning the repeated correction.

But the legacy fast controller could then **erase the good prior**. A fixed probe radius / step designed for a 0.6 residual became destructive once slow memory had moved the state close to the solution.

The fix did not require a new biological mechanism. Conventional optimizer hygiene was enough:

```text
1. infer rough residual distance from the noisy known relevance curve
2. shrink probe radius / step near high relevance
3. spend one extra scalar measurement on the proposed fast update
4. reject the update if measured relevance does not improve
```

Dense 128-D, consequence noise 0.01:

```text
                              late start   late final   late probes   success
Kalman + scaled/accept          0.164        0.164          30.7      95.8%
EMA + scaled/accept             0.162        0.162          24.2      93.3%
frozen + scaled/accept          0.606        0.177         411.3      63.3%
shuffled + scaled/accept        0.666        0.191         463.3      38.3%
```

Sparse4 gives the same qualitative result:

```text
Kalman + scaled/accept          32.5 probes   94.2% success
EMA + scaled/accept             39.9          90.8%
frozen                         345.5          80.8%
shuffled                       415.9          50.0%
```

At noise 0.01, the learned systems save roughly **89--94%** of their first-round scalar probe cost. Most late episodes are already inside the target radius before any directional probe.

At consequence noise 0.02 the benefit survives but noisy stopping becomes the next boundary. Kalman + scaled/accept still saves about 71% of probes in both dense and sparse worlds, while frozen and shuffled controls remain near the full 512-probe budget.

Gate 9 therefore earns:

> **Repeated noisy fast corrections can become context-specific slow priors that drastically reduce future search cost, provided the fast residual optimizer becomes cautious near a learned solution.**

It does **not** earn a novel slow learner. Ordinary EMA and Kalman-style estimation are sufficient.

It does earn two architectural roles:

- correctly addressed slow memory;
- a fast/slow handoff that prevents exploratory dynamics from overwriting a good slow prior.

Canonical evidence:

- [SLOW_CONSOLIDATION_FORK.md](SLOW_CONSOLIDATION_FORK.md)
- `experiments/fork_slow_consolidation_noise.py`
- `experiments/fork_slow_fast_handoff.py`
- `results/fork_slow_consolidation_summary.json`

## Gate 10 — transfer across changed coordinates

Gate 9 still gave each context an explicit memory row in one fixed latent basis. Gate 10 attacks that loophole directly.

The world contains a rank-4 family of hidden context corrections embedded in a 32-D observed latent space. Each view renders the same hidden correction family through a different random orthonormal basis.

The fast stage is collapsed to a noisy correction packet so this gate isolates the representation/alignment problem rather than re-testing Gate 8/9 search.

Three views are seen during training. A fourth basis is held out completely.

With **zero calibration** on the unseen view:

```text
blind shared rendered table     error 0.684   success 0%
explicit context x view table   error 0.600   success 0%
ordinary ridge                  error 0.600   success 0%
oracle exact basis              error 0.014   success 100%
```

This establishes an identifiability boundary rather than an architectural failure:

> **An arbitrary unseen basis cannot be decoded from no relation signal at all.**

Then the new view is given a few calibration contexts whose identity is shared with the learned reference. An ordinary linear map is fitted and tested on the remaining contexts.

The hidden family has rank 4:

```text
paired calibration contexts     ridge error    success    cosine

0                                  0.600         0.0%      0.000
2                                  0.358        11.7%      0.736
3                                  0.247        37.3%      0.879
4                                  0.173        63.4%      0.942
6                                  0.108        90.3%      0.981
8                                  0.088        97.9%      0.989

oracle exact basis                0.014       100.0%      1.000
```

Orthogonal Procrustes also works, but ordinary ridge is stronger under the noisy finite calibration used here. A simpler view-0 reference performs essentially as well as the more elaborate alignment of all three training views.

So Gate 10 does **not** earn a novel alignment algorithm.

It earns a clean representation result:

> **Slow knowledge can transfer across changing latent bases once the machine has enough information to relate the coordinate systems. For a low-rank correction family, a small set of cross-view correspondences lets ordinary linear alignment re-render the learned correction into an unseen basis.**

Canonical evidence:

- [CROSS_BASIS_FORK.md](CROSS_BASIS_FORK.md)
- `experiments/fork_cross_basis_transfer.py`
- `results/fork_cross_basis_transfer_summary.json`

This is the first direct meeting point between the current fast/slow architecture and the earlier Tuesday matrix/source-separation line.

## Gate 11 — unlabeled temporal alignment

Gate 10 still supplied explicit cross-view context correspondences. Gate 11 removes them.

Two views now contain independent trajectories generated by the same latent process family and rendered through different random bases. There are:

- no synchronized samples;
- no context-pair labels;
- no direct cross-view regression target.

The question is whether dynamics themselves contain enough information to reconstruct a usable coordinate relation.

### Sawtooth world — temporal identity + temporal polarity

Four hidden components have nearly identical symmetric marginals but different periods.

Static FastICA can recover the source axes, but cannot by itself determine which axis in view A corresponds to which axis in view B or which way each axis points.

```text
method                         axis recovery   transfer error   success

PCA / zero-lag                    ~0.70            0.855          0.9%
FastICA static                    ~1.00            0.766          4.4%
FastICA + temporal match          ~1.00            0.751         10.4%
FastICA + temporal match
        + temporal orientation    ~1.00            0.011        100.0%

multi-lag temporal basis          ~0.999           0.773         10.9%
multi-lag + temporal orientation  ~0.999           0.029        100.0%
```

The key decomposition is:

```text
autocorrelation profile
    -> component identity

sign-sensitive increment asymmetry
    -> component orientation
```

This is not a SOBI-specific victory. FastICA plus the same temporal matching/orientation is slightly better in this non-Gaussian world.

So Gate 11 earns a more general statement:

> **Dynamics can provide the missing correspondence information between independently sampled coordinate systems: temporal fingerprints identify components, and sign-sensitive temporal asymmetry can orient them.**

### Gaussian AR control — axes without orientation

The Gaussian AR world removes static non-Gaussianity. At one instant the latent state is approximately isotropic Gaussian, but the components have distinct autocorrelation time constants.

```text
method                         axis recovery   transfer error   success

PCA / zero-lag                    ~0.70            0.838          0.2%
FastICA static                    ~0.79            0.814          0.5%
multi-lag temporal basis          ~0.999           0.801          9.2%
multi-lag + sign heuristic        ~0.999           0.741         17.0%
```

The temporal decomposition recovers the source **axes** almost perfectly.

But a zero-mean Gaussian AR source is exactly invariant under:

```text
s_i(t) -> -s_i(t)
```

Therefore independent unpaired observations contain no information that can determine the relative sign between views.

That is an identifiability boundary, not a failed optimizer:

> **Second-order temporal structure can identify an axis without identifying its orientation.**

### Shuffled-time control

Destroying temporal order kills the positive result:

```text
sawtooth with time shuffled independently

FastICA + temporal signed     error 0.858   success 0.3%
multi-lag temporal signed    error 0.791   success 0.7%
```

So the alignment genuinely depends on dynamics.

Canonical evidence:

- [UNLABELED_ALIGNMENT_FORK.md](UNLABELED_ALIGNMENT_FORK.md)
- `experiments/fork_unlabeled_temporal_alignment.py`
- `results/fork_unlabeled_temporal_alignment_summary.json`

## Gate 12 — scalar consequence resolves residual sign bits

Gate 11 reduced the symmetric Gaussian cross-basis problem to a small exact ambiguity:

```text
arbitrary basis relation
        ↓ temporal decomposition
axes + component identity
        ↓
one unresolved sign bit per component
```

For rank 4 that means only 16 possible orientation maps.

Gate 12 asks whether KyberDyyni's existing local scalar consequence can finish that job without restoring explicit cross-view labels.

The answer is yes, and the winning update rule is deliberately ordinary.

### Bitwise consequence attack

For each unresolved component, compare two candidate maps that differ only in that sign bit. Accumulate the scalar relevance difference across unrelated calibration contexts.

No vector-valued B-side correction is exposed.

```text
no consequence noise

method                    contexts   scalar evals   sign accuracy   transfer success

random signs                  --           0           52.1%             15.8%
hill climb                     1           5           96.4%             88.3%
bitwise                        1           8           95.8%             86.6%
bitwise                        2          16          100.0%            100.0%
exhaustive 16 patterns         2          32          100.0%            100.0%
shuffled consequence          32         256           51.6%             13.8%
```

The exact sign oracle's transfer floor is about:

```text
error   0.0376
success 100%
```

and ordinary bitwise accumulation reaches that floor.

### Consequence noise

More contexts supply repeated evidence naturally:

```text
consequence noise      contexts to oracle floor      scalar evals

0.00                               2                         16
0.01                               4                         32
0.02                               8                         64
0.04                              16                        128
```

At sigma 0.04:

```text
K=4    success 96.6%   sign accuracy 99.0%
K=8            98.3%                 99.5%
K=16          100.0%                100.0%
```

Repeatedly measuring the same +/- pair twice is not worth its doubled cost here. Different contexts are the better repetitions because they excite different components with different strengths.

The shuffled-consequence control stays at chance sign accuracy and poor transfer even after 256 scalar evaluations.

So Gate 12 earns:

> **Blind temporal alignment can compress an arbitrary coordinate problem into a handful of global binary ambiguities, and ordinary local scalar consequence can resolve and retain those bits for future transfer.**

It does **not** earn a novel sign learner.

Canonical evidence:

- [SIGN_CONSEQUENCE_FORK.md](SIGN_CONSEQUENCE_FORK.md)
- `experiments/fork_sign_consequence_calibration.py`
- `results/fork_sign_consequence_calibration_summary.json`

## Gate 13 — rank scaling and weak excitation

Gate 12 solved the residual sign ambiguity at rank 4. Gate 13 asks whether scalar-consequence calibration still behaves sensibly at rank 8 / 16 / 32, and whether every unresolved component must be probed on every context.

The fork conditions on Gate 11 already having recovered the source axes and component identity. It isolates only the final orientation-calibration cost.

Three context worlds were tested:

- dense: every context excites every component;
- sparse4: exactly four random components are active;
- heavy-tail: all components exist, but later components are progressively weaker.

Consequence noise was 0.02 and 0.04.

### Rank 32 — selective measurement beats full probing

Dense, noise 0.02, 64 calibration contexts:

```text
method           transfer success    error       scalar evaluations

full bitwise          99.13%         0.0035            4096
top-4                 99.17%         0.0034             512
active-4              99.19%         0.0035             512
random-4              19.72%         0.3648             512
shuffled full          0.00%         0.8337            4096
```

The same eight-scalar-per-context ceiling reaches essentially the full solution with **8x fewer consequence measurements**.

Sparse excitation makes the reason explicit:

```text
rank 32 / sparse4 / noise 0.02 / 64 contexts

full bitwise    97.22% success   4096 evals
top-4           99.11%            512
active-4        98.92%            512
```

Twenty-eight inactive components provide zero signal but still inject consequence noise if they are probed. Measuring fewer coordinates can therefore be better than measuring all of them.

### Heavy-tail world — exploration matters

Always taking the four strongest currently expressed components neglects weak freedoms indefinitely.

A minimal exploration discount,

```text
priority_i = |current coefficient_i| / sqrt(1 + times_probed_i)
```

keeps weak under-tested components alive.

At rank 32 / noise 0.04 / 128 contexts:

```text
                    success     scalar evaluations

full bitwise          79.7%          8192
top-4                 59.0%          1024
active-4              81.7%          1024
```

Active-4 slightly beats the full O(R)-per-context attacker while using one eighth the scalar consequence budget.

Heavy-tail worlds also expose a metric problem: raw sign accuracy can be only ~82% while energy-weighted sign accuracy is already ~98.5%. The unresolved bits are mainly weak freedoms that barely affect behavior.

Gate 13 therefore earns:

> **A finite consequence budget should be allocated to the degrees of freedom that are informative now, while preserving exploration of under-tested freedoms. The relevant scaling variable is not rank alone but rank × excitation × consequence SNR.**

No novel learning rule is claimed. The winning mechanism is ordinary active measurement allocation.

Canonical evidence:

- [RANK_SCALING_FORK.md](RANK_SCALING_FORK.md)
- `experiments/fork_rank_scaling_sign.py`
- `results/fork_rank_scaling_sign_summary.json`

## Next gate

### Gate 14 — temporal separator scaling and signature crowding

Gate 13 assumes the upstream temporal stage already recovered all source axes and component identities.

That assumption must now be attacked.

The question is:

> **Does unlabeled temporal alignment survive higher rank, finite observation windows, noise, and increasingly similar temporal signatures?**

Gate 14 should vary:

- rank: 4 / 8 / 16 / 32;
- observation length;
- stream noise;
- spacing between temporal time constants / periods;
- exact degeneracy where two components have identical temporal statistics.

Attackers:

- PCA / zero-lag covariance;
- one-lag AMUSE;
- multi-lag SOBI-style decomposition;
- FastICA where non-Gaussian marginals exist;
- oracle true source axes;
- shuffled-time control.

The critical metric is component identity recovery across independent views, not merely reconstruction of the latent subspace.

A required control is exact temporal degeneracy:

> If two components have identical second-order temporal statistics, the method must report or exhibit rotational ambiguity inside that subspace. A method that confidently invents a unique orientation there has failed the test.

If performance is governed mainly by **separation between dynamical signatures relative to finite-window estimation noise**, that is a much more useful scaling law than “high dimension is hard.”

## Kill conditions

KyberDyyni should be considered unnecessary if ordinary alternatives win cleanly:

- current-state MLP;
- explicit delay features;
- explicit phase/slot counter;
- reservoir + linear readout;
- GRU/RNN;
- simple Markov statistic;
- ordinary attention;
- hand-coded search;
- finite-difference / coordinate probing;
- SPSA and other zeroth-order optimizers;
- ordinary EMA / online estimation for slow consolidation;
- ordinary coordinate alignment / regression for cross-basis transfer;
- ordinary covariance / CCA alignment for unlabeled cross-view transfer;
- ordinary binary evidence accumulation for sign orientation;
- ordinary active measurement allocation for rank-scaled consequence probing.

The point is not to make biology-shaped software. The point is to see whether separating **fast sampling**, **trajectory-relative addressing**, **slow consolidation**, and **coordinate relation** gives us a useful machine that ordinary static-weight thinking obscures.
