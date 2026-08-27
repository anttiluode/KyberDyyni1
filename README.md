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

The current `ThetaScanner` is deliberately simpler than their continuous-attractor model. Its left/right alternation is engineered. One of the first serious attacks is to replace that hard-coded alternation with an adaptation-generated moving attractor.

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
fast control before slow learning       ~0.893
no cue before consolidation             ~0.502
no cue after slow consolidation         ~0.891
```

This is deliberately a small result:

> **Fast state can solve an immediate problem before slow weights know it; repeated delayed local teaching can later consolidate the mapping.**

The current gate does **not** prove that phase tagging is necessary. A shuffled short address queue performs nearly as well because neighboring samples often share the same context. That failed attack is informative: Gate 2 must create a task where the same context contains multiple phase-specific candidate events so delayed consequence is ambiguous without the phase coordinate.

## Phase hypothesis

`PhaseBridge` emits a local coordinate containing:

- anchor;
- probe displacement;
- sweep side;
- fast axis/focus state;
- `sin(theta)` / `cos(theta)` and a second harmonic;
- context;
- phase × context conjunctions.

This is an engineering hypothesis:

> **Phase may be useful as part of the address of a fast internal event, allowing slow local plasticity to distinguish different moments of an internally generated trajectory.**

It is **not** yet an earned result.

## No backpropagation

There is no PyTorch, JAX, TensorFlow, autograd, gradient tape, reverse graph traversal, or BPTT.

The slow learner uses a delayed local address and a bounded delta-like structural update:

```text
local context/address
        |
        +---- retained briefly
                 |
             consequence
                 |
                 v
    local structural update
                 |
        finite L1 budget
```

This still uses a teaching/consequence signal. It is not a solution to arbitrary deep credit assignment.

## Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_theta_scanner.py
python experiments/gate1_fast_slow_consolidation.py
python run_all.py
python -m unittest discover -s tests
```

## Next gates

### Gate 2 — phase must matter

Construct one continuous context in which several candidate internal samples occur during every sweep. Deliver delayed consequence that identifies only one part of the trajectory.

Pass condition:

```text
intact phase address >> shuffled phase address
```

If an ordinary delay queue or current-state statistic solves it equally well, phase has not earned a job.

### Gate 3 — adaptation generates the sweep

Replace hard left/right cycle parity with a small continuous attractor whose bump motion emerges from:

- recurrent attraction,
- firing-rate adaptation,
- periodic modulation.

Attack it with the simpler engineered alternator.

### Gate 4 — useful internal search

Put a target in a small latent world. The anchor represents "where the system is." The fast scanner samples nearby alternatives.

Test whether relevance can:

- redirect the axis;
- narrow the sector;
- increase sampling frequency;
- improve decisions before slow weights change.

### Gate 5 — consolidation from self-generated samples

Turn off the external fast cue. Let internally generated sweeps produce candidate trajectories. Only later consequence is available.

Ask whether useful trajectories slowly become preferred by structural learning.

### Gate 6 — offline / REM-like mode

Remove external input while keeping the internal scanner alive. Replay or generate trajectories from slow anchors and test whether offline sampling changes later online behavior.

This is an engineering test of a shared online/offline sampler, not a model of REM.

## Kill conditions

KyberDyyni should be considered unnecessary if ordinary alternatives win cleanly:

- current-state MLP;
- explicit delay features;
- reservoir + linear readout;
- GRU/RNN;
- simple Markov statistic;
- ordinary attention;
- hand-coded search.

The point is not to make biology-shaped software. The point is to see whether separating **fast sampling**, **phase-relative addressing**, and **slow consolidation** gives us a useful machine that ordinary static-weight thinking obscures.
