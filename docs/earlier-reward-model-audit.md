# Earlier study: reward-model selection of tool trajectories

This is the **37-pool / 370-candidate study referenced in my CV**. The
current local manuscript title is *When Reward Models Choose Tool
Trajectories: An Auditable Study of Utility, Serialization, and Graph
Sensitivity*. An earlier title was *A Pool-Level Audit of Selected
Reward-Model Checkpoints for Tool-Trajectory Selection*.

It remains an ongoing, unpublished research manuscript; no venue acceptance
is claimed here. The later WebArena task-evidence direction is a separate
study, described in the [main README](../README.md). Its 163 development
tasks are not additional pools or a replication of this earlier audit.

## Question and protocol

When a selected reward model ranks executable tool-use trajectories, does
score repeatability also give stable and useful winner selection?

The earlier primary natural analysis retains **37 mixed-outcome tau²-bench
Airline task pools, ten candidates per pool, or 370 candidate trajectories**.
Its analysis separates score-defined top sets, tie resolution, executable
selection success, and protocol sensitivity. Pool/task is the independent
unit; five repeats or three serializations do not multiply that sample size.

The evaluator is a locally trained **Qwen3-4B-Instruct-2507** model with
LoRA adapters and a two-class classification head, jointly trained on
800 pointwise and 232 pairwise examples. The primary checkpoint used seed
20260920 and was selected at epoch 3 by joint validation negative
log-likelihood. Its base-model revision is
`cdbee75f17c01a7cc42f958dc650907174af0554`.

The repeatability and serialization results below use its **frozen
pointwise interface**: complete trajectory JSON is scored through a
classifier forward pass, and the difference between the two logits is
temperature-calibrated into a success score. These results do not evaluate
the combined pointwise/pairwise selection policy. They concern one selected
evaluator in the frozen September 2026 protocol, rather than every reward
model or a ranking of evaluator classes.

## Findings traced to retained analysis

| Check | Retained result | Meaning |
|---|---|---|
| Five separately invoked deterministic processes | Same selected winner in **37/37 pools**; reported maximum score drift **0** | Repeatable under the fixed process and representation |
| Three semantically equivalent JSON representations | Same winner in **21/37 pools (56.8%)**; 15 pools have two distinct winners and one has three | Selection can change when representation changes |
| Score association across those representations | Mean Spearman correlation **0.9685**; maximum within-pool rank displacement **7** | High overall score correlation does not ensure the same selected trajectory |
| Serialization-specific selection success | **21.6%, 21.6%, 29.7%** across the three recorded conditions | Utility is conditional on the protocol; this is not evidence of a validated mitigation |

The three conditions are `sorted-compact-v1`, `insertion-compact-v1`, and
`sorted-pretty-v1`. The broader manuscript also examines sparse
graph/orientation sensitivity and native-context constraints; those larger
records are outside this compact presentation release.

## Inspect the compact evidence

[`data/earlier_audit_summary.json`](../data/earlier_audit_summary.json)
contains only aggregate metadata, anonymous histogram counts, and selected
reported summaries. Original pool IDs, candidate IDs, trajectory content,
candidate scores, model outputs, adapters, and reviewer materials are absent.

From the repository root:

```bash
python scripts/check_earlier_audit.py
```

The CPU check sums the histogram denominators, reconciles 37×10=370,
and recomputes **21/37** serialization agreement and **37/37** process
agreement. It checks the reported condition-level rates against the stated
denominator. It does not rerun model scoring, reconstruct candidate-level
winners, or independently verify the reported zero score drift.

## Sources and limits

The extraction was checked against the actual local supplementary records,
not only the manuscript abstract:

- `ars-gpu-closure-20260920/attempt-gate4-analysis-formal-r1/`
  `serialization_per_pool.jsonl` and `repeatability_per_pool.jsonl`:
  the histogram counts are computed from each record's selected-winner
  equality, then all identifying fields are discarded.
- The same directory's `serialization_stability.json` and
  `repeatability.json`: conditions, candidate/pool counts, score summaries,
  and selection-success summaries.
- All three serialization completion records and all five repeat completion
  records: 370 rows, 37 mixed pools per attempt; the original analysis
  `matrix_stability` explicitly requires ten candidates per pool.
- `paper/eswa2026/results/serialization_policy_metrics.csv`: separate
  local reanalysis table agreeing with the retained condition summaries.
- `scripts/train_ars_joint_evaluator_20260920.py`, its retained training
  and dataset records, and `scripts/run_ars_joint_evaluator_20260920.py`:
  model identity, checkpoint selection, and the actual classifier interface.

These source files remain in the private research workspace. Their relative
names identify provenance; they are not links to files bundled here.

The primary natural source is one airline dataset and selected checkpoints.
Training examples were limited to 8,192 tokens, whereas the retained
repeatability run scored complete inputs up to 125,301 tokens. Long-context
scoring is therefore outside the observed training-length support; the
temperature calibration does not establish reliability at those lengths.
The wider manuscript contains coverage-restricted and internal diagnostics;
the AgentSuite snapshot has no sufficiently documented upstream accession
and redistribution record, so it is not presented as a public reproducible
benchmark. Repeated-process agreement is not invariance to representation,
and this audit does not establish universal reward hacking, causality,
online-RL improvement, or a broadly effective remedy.
