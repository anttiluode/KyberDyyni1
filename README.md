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

The current `ThetaScanner` is deliberately simpler than their continuous-attractor model. Its left/right alternation is engineered. Gate 3 attacks that simplification.

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
python run_all.py
python -m unittest discover -s tests
```

Results are saved in `results/`.

## Next gates

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

There is no `left`, `right`, cycle parity, or alternating-sign instruction in this scanner. The full system spontaneously settles into large bounded sweeps that reverse side every theta cycle.

This earns:

> **A fast alternating sampler can be generated by continuous population dynamics rather than imposed as an external sequence. Adaptation creates mobility, recurrence supports a coherent bump, and theta organizes that mobility into a stable rhythm.**

It does **not** show that this population is more useful than the tiny engineered `ThetaScanner`. The engineered scanner remains dramatically simpler and alternates perfectly by construction. Gate 4 must make the dynamical scanner perform useful search before the added mechanism has earned its engineering cost.

### Gate 4 — useful internal search

Put a target in a small latent world. The anchor represents "where the system is." The fast scanner samples nearby alternatives.

Test whether relevance can:

- redirect the axis;
- narrow the sector;
- increase sampling frequency;
- improve decisions **before** slow weights change.

Attack with hand-coded search and ordinary attention.

### Gate 5 — consolidation from self-generated samples

Remove the external fast cue. Let internally generated sweeps produce candidate trajectories. Only later consequence is available.

Ask whether useful internally sampled trajectories slowly become preferred by structural learning.

### Gate 6 — offline / REM-like mode

Remove external input while keeping the internal scanner alive. Replay or generate trajectories from slow anchors and test whether offline sampling changes later online behavior.

This is an engineering test of a shared online/offline sampler, not a model of REM.

### Gate 7 — coupled fast and slow populations

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
