# Task Evidence for Tool-Trajectory Evaluation

Research by **Junxi Tan** on evaluating tool-using agents from the evidence
in their recorded trajectories. This snapshot contains the current
task-evidence implementation, a synthetic example, and compact development
results that can be reanalyzed on a CPU.

**Question:** Does an explicit record of which task conditions depend on
which observed entities, attributes, and operations improve trajectory
selection over a strong natural-language evidence check?

The current development experiment gives a **negative result** for the
implemented structured representation: **44.48%** expected selection
success versus **46.93%** for the natural-language baseline. The paired
difference is **−2.454 percentage points**, with a template-cluster
bootstrap 95% interval of **[−5.257, −0.095]**. This is development evidence,
not a held-out or independently reproduced model result.

## What is implemented

- A chronological renderer for recorded browser-agent trajectories. It
  removes repeated cumulative-message packaging while preserving every
  current-page observation and explicitly recording missing source input.
- Five evaluator protocols: full record (A), terminal state (B), latest-field
  table (C), goal-by-goal natural-language evidence (D), and structured
  dependency evidence (E). The shared goal conditions and final scoring
  request keep the target task consistent across the comparison.
- Sequential processing of long records with carried evidence notes;
  explicit source-step references, relevant later changes, recovery, and
  unknown support. These are evaluator instructions, not a proven guarantee
  that the model follows them correctly.
- Task-level expected success under uniform tie resolution, whole-pool
  fallback for missing scores, and task-paired inference with resampling
  clustered by task template.

The original runner and its prompt definitions are in
[`scripts/run_eswa_task_evidence.py`](scripts/run_eswa_task_evidence.py).
The [study configuration](data/study_config.json) records the evaluated
model, budgets, arms, and primary comparison.

For C/D/E, the final scorer receives the **complete original transcript
plus the derived note** on the full route, or the **last raw chunk plus
the accumulated note** on the chunked route. The table or structured
record supplements that visible input; it does not replace it entirely.

## Current evidence

The earlier **37-pool / 370-candidate reward-model audit** referenced in
my CV has a [separate evidence guide](docs/earlier-reward-model-audit.md).
The **163-task WebArena study below is the later direction**; the two
experiments have separate questions, protocols, and denominators.

The completed `dev-qwen4b-a02` experiment used
**Qwen3-4B-Instruct-2507**, BF16, greedy decoding with thinking disabled,
and vLLM 0.11.0 on one RTX 5090 32 GB. It retained **163 development
tasks, 38 template groups, and 489 candidate trajectories**. All **9,910
model calls** completed and all **2,139 final scores** parsed.

| Method | Evaluated tasks | Expected selection success |
|---|---:|---:|
| Uniform selection | 163 | 43.76% |
| B: terminal state and final answer | 163 | 47.34% |
| C: latest entity/field table | 163 | 46.32% |
| D: natural-language evidence check | 163 | 46.93% |
| E: dependency evidence records | 163 | 44.48% |
| Pool oracle | 163 | 58.90% |

Method A was evaluated only on **61 complete-context tasks** and reached
47.81%; its denominator differs from the table. There are **50
mixed-outcome pools** among the 163 tasks. The sole designated primary
contrast is E−D; the other point estimates do not establish a ranking.

The E−D analysis has eight improved and 14 harmed tasks, with a net change
of −4.0 expected successes. Evidence-note truncation occurred in 116 E
calls and 15 D calls, although final scores all parsed. This is a diagnostic
observation and does not establish the cause of the negative result.

The **647-task held-out split has not been scored** in the retained local
record. The 8B evaluation, planned ablations, new serialization study, and
SWE-Gym extension also have no completed results in this snapshot. The
evidence status remains **ANALYZED**. See
[`docs/research-status.md`](docs/research-status.md) for the dated scope
and the distinction from the earlier reward-model audit.

## Run the CPU example and reanalysis

Use Python 3.10–3.13 and NumPy. No GPU, model download, network access,
or external credentials are needed for these commands.

```bash
python -m pip install -r requirements.txt
python scripts/demo.py
python scripts/reproduce.py
python scripts/check_earlier_audit.py
python -m unittest discover -s tests -v
```

`demo.py` runs the real renderer and prompt builder on a clearly labelled
synthetic browser record. It does not call a model or report benchmark
performance. `reproduce.py` checks task/group denominators and the released
713 task-arm utility rows, then recomputes the task-level means and the
10,000-draw template-cluster bootstrap used for E−D. A successful run prints
`PASS` and the reproduced result.

This reproduces the **analysis of retained outputs**, not the original
model inference. The runner's GPU `prepare` and `run` modes require
separately obtained datasets, source-format inputs, local model weights,
and the original inference environment; those assets are not bundled.

## Files and provenance

| Path | Contents |
|---|---|
| [`scripts/run_eswa_task_evidence.py`](scripts/run_eswa_task_evidence.py) | Original current research runner: rendering, prompts, model interface, and analysis |
| [`scripts/demo.py`](scripts/demo.py) | Synthetic renderer and prompt example |
| [`scripts/reproduce.py`](scripts/reproduce.py) | Read-only CPU reanalysis and logical checks |
| [`docs/earlier-reward-model-audit.md`](docs/earlier-reward-model-audit.md) | Earlier 37-pool study, CV alignment, findings and limitations |
| [`data/earlier_audit_summary.json`](data/earlier_audit_summary.json) | Anonymous winner-agreement histograms for the earlier study |
| [`data/development_pool_utilities.jsonl`](data/development_pool_utilities.jsonl) | Task-arm expected utilities; consistently renamed task/group IDs |
| [`data/development_summary.json`](data/development_summary.json) | Retained development aggregates and primary contrast |
| [`data/study_config.json`](data/study_config.json) | Evaluated settings and pending experiments |
| [`docs/provenance.md`](docs/provenance.md) | Source versions, extraction boundary, and external dependencies |

This is a curated research snapshot. It excludes raw browser observations,
task text, individual model responses, per-candidate labels, model weights,
private research history, reviewer correspondence, and submission drafts.
The public aggregate rows preserve the pairing and clustering structure
needed for the reported statistical analysis.

## Contact and reuse

Junxi Tan · Guangdong University of Finance and Economics ·
[GitHub](https://github.com/TanC8186)

This is ongoing research; no acceptance or publication is claimed for this
study. See [RIGHTS.md](RIGHTS.md) for the ownership and reuse statement and
[`docs/provenance.md`](docs/provenance.md) for third-party references.
