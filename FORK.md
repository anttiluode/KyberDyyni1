# Control-law fork

This branch asks the Vollan-inspired focused-sampling question many different ways before selecting an engineering interpretation.

All controllers share the same fast relevance signal and use **no slow learning**.

The bakeoff separates:

- moving the generator's external anchor;
- keeping the external reference fixed while displacing a downstream sampling coordinate;
- theta-rate modulation only;
- downstream angular-sector compression only;
- coupled rate/width under an approximate fixed sampling budget;
- broad-while-uncertain / contract-when-confident sampling;
- downstream relevance gating;
- one-side suppression;
- learned phase-priority weighting;
- the original naive adaptation+tether+rate translation;
- random dither;
- no-relevance control.

Every controller is evaluated in three worlds:

1. stationary target acquisition;
2. continuously moving pursuit;
3. target loss followed by an abrupt reorientation.

The purpose is not to tune one winner. The question is whether any artificial translation of direction/width/frequency control survives multiple worlds while retaining the stable-reference / continuous-sampling properties that motivated the fork.

Canonical `main` remains untouched until this branch produces a result worth merging.
