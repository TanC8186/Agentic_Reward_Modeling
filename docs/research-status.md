# Research status

This snapshot was prepared on November 16, 2026 from the current local
research workspace. The latest retained completed development record is
dated November 8, 2026. No remote server was queried for a newer experiment
as part of this portfolio preparation.

## Current task-evidence direction

The active implementation evaluates whether explicit task-dependent
evidence records help select successful browser-agent trajectories.
Development run `dev-qwen4b-a02` completed at 04:19 on November 8, 2026
(Asia/Shanghai): 163 tasks, 38 template groups, 489 candidates, 9,910 calls,
and 2,139 parsed final scores. Its analysis is negative for E relative to D.

The earlier `dev-qwen4b-a01` attempt was stopped because historical agent
output instructions contaminated the evaluator's response format before
final utility results were available. It is not a method-comparison result.
The a02 implementation restates the evaluator role and each method's
requested output after the historical record for every method.

The full 163-task development set is retained in the a02 analysis. Task
templates define bootstrap clusters; candidate trajectories and repeated
model calls are not additional independent tasks. Ties are evaluated as
uniform expected selection, rather than resolved with outcome labels.
For C/D/E, the final scorer on a full-route task sees the original complete
transcript together with the derived evidence note. On a chunked-route task
it sees the final raw chunk and the note accumulated while reading all
chunks. The comparison therefore changes the evidence-note protocol within
this routing design; it is not a pure replacement of raw input by a table
or schema.
The development result is not a held-out confirmation or a claim that
dependency evidence can never help under another implementation.

The local record reports that 647 held-out tasks are prepared
but unscored. The 8B evaluation, E_time/E_static ablations, new serialization
evaluation, and SWE-Gym extension remain pending. An eight-pool ToolRM
development diagnostic completed separately, but its six all-success and
two all-failure pools cannot distinguish selection methods; its 75% success
must not be used as a competitive baseline for the 163-task result.

## Earlier reward-model audit

An earlier manuscript studies checkpoint selection utility, semantic
serialization, and pairwise graph/orientation sensitivity on 37 mixed-outcome
tau²-bench Airline pools (370 candidates). The current local manuscript
title is *When Reward Models Choose Tool Trajectories: An Auditable Study of
Utility, Serialization, and Graph Sensitivity*.

That manuscript's abstract still summarizes the earlier 37-pool study.
The subsequent 163-task WebArena development direction has its own code and
dated result records; it is not a new denominator for the earlier audit.
This public snapshot focuses on the later task-evidence implementation and
includes a [compact guide to the earlier audit](earlier-reward-model-audit.md)
to match the CV. It does not redistribute either submission manuscript or
reviewer materials.
No venue acceptance is claimed for either ongoing line in this snapshot.
