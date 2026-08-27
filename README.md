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
      FAST THETA SCANNER H(t)
      - continuous phase
      - left/right sampling
      - fast retargeting
      - width modulation
      - frequency modulation
            |
            v
      PHASE BRIDGE E(t)
      - where in the sweep?
      - relative to which anchor/context?
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
2. **Phase/context address** — binds an internal sample to its location in an ongoing trajectory.
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

## Next gates — consolidated line

The response-comparison sibling independently converged on a useful missing experiment: **do not jump directly from "the sweep can search" to "slow learning." First let relevance alter the fast sweep itself, then ask whether repeated successful fast searches become slow priors.** That idea is now part of the canonical roadmap; its local code was never committed.

### Gate 5 — fast relevance -> slow prior

Gate 5 joins the pieces that Gates 0–4 tested separately:

```text
context / stable anchor
        |
        v
self-generated sweep
        |
scalar relevance arrives during the sweep
        |
        +--> fast retarget / narrow / accelerate
        |        (no slow weight change yet)
        |
later consequence
        |
phase/context-local trace
        |
        v
bounded slow prior
        |
next encounter starts closer to where experience says to search
```

This gate has two required halves.

**5A — fast modulation before learning.** Relevance must change direction/sector/frequency immediately while slow weights are frozen. This is the Vollan-shaped part that the current adaptation scanner has not yet earned.

**5B — consolidation after success.** Repeated successful searches must alter bounded slow structure so that a later recurrence of the same context begins with a better prior even when the original fast relevance cue is absent.

Attackers must include the current engineered sweep, random dither, a simple context->target exponential moving average / bandit prior, and a frozen-slow ablation. If an ordinary EMA prior plus random search wins cleanly, the fancy mechanism loses.

The proposed experimental contract is written in [GATE5.md](GATE5.md).

### Gate 6 — continuity must matter

Gate 4 exposed the adaptation scanner's only current advantage over random dither: far lower internal travel. Make that property consequential rather than cosmetic.

Use worlds where probe-to-probe transitions carry state, cost, hysteresis, eligibility, or other path dependence. Then ask whether a continuous sweep beats equally budgeted random proposals.

If path continuity does not improve anything, the attractor machinery remains unnecessary.

### Gate 7 — offline / REM-like mode

Remove external input while keeping the internal scanner alive. Replay or generate trajectories from slow anchors and test whether offline sampling changes later online behavior.

This is an engineering test of a shared online/offline sampler, not a model of REM.

### Gate 8 — coupled fast and slow populations

Only after the small gates survive: make the fast scanner and slow structural population recurrently interact rather than remain cleanly separated modules.

## Kill conditions

KyberDyyni should be considered unnecessary if ordinary alternatives win cleanly:

- current-state MLP;
- explicit delay features;
- explicit phase/slot counter;
- reservoir + linear readout;
- GRU/RNN;
- simple Markov statistic;
- ordinary attention;
- hand-coded search.

The point is not to make biology-shaped software. The point is to see whether separating **fast sampling**, **phase-relative addressing**, and **slow consolidation** gives us a useful machine that ordinary static-weight thinking obscures.
