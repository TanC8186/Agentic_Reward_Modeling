# Provenance and reproduction boundary

## Source material

The research runner is copied from the current local
`scripts/run_eswa_task_evidence.py`, version used by the
`eswa-task-evidence-v1-20261107` development study. The runner contains the
original evidence prompts, renderer, long-record routing, model interface,
probability parser, and template-cluster bootstrap. Its inference imports
are lazy, so importing the renderer or analysis does not load model weights.

The released numeric evidence is extracted from
`dev-qwen4b-a02/pool_selections.jsonl`, `results.json`, and `completion.json`.
It contains 713 rows: A on 61 complete-context tasks and B–E on 163 tasks
each. Task IDs and template-group IDs are consistently renamed in original
encounter order. Raw candidate identifiers, site names, page observations,
task text, per-candidate labels, model answers, and source-machine paths
are excluded. The extraction preserves utility, random-selection utility,
oracle utility, route, top-set size, mixed-outcome status, and fallback
status. A task's grouping and pairing are unchanged.

The snapshot's CPU script recomputes descriptive means and the primary
task-paired, template-cluster bootstrap directly from those rows with the
original analysis function. It does not regenerate candidate scores or
independently verify the upstream executable outcome evaluation. The
logical checks are denominator/schema/value checks, not proof of byte
identity or independent end-to-end inference reproduction.

The experiment completion record labels model output `UNVERIFIED`; the
retained postprocessing result is `ANALYZED`. Those labels are preserved
as development evidence rather than promoted by packaging checks.

## Evaluated settings

- Qwen/Qwen3-4B-Instruct-2507; BF16; greedy; thinking disabled.
- vLLM 0.11.0; RTX 5090 32 GB; maximum batch size eight.
- 32,768-token operating context; 24,000-token full-record routing threshold;
  8,000-token evidence chunks; 1,024-token evidence-note limit;
  256-token final-score limit.
- Seed 20261107; 10,000 template-cluster bootstrap draws for E−D.

The CPU release requires only NumPy. GPU inference additionally depends on
PyTorch, Transformers, vLLM, externally obtained model weights and input
records. Full model re-execution is outside this compact release's tested
reproduction tier.

## Third-party references

- WebArena: <https://github.com/web-arena-x/webarena> — browser-task benchmark
  used by the later task-evidence study; no raw benchmark records included.
- Qwen3: <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507> — evaluated
  checkpoint; no weights or tokenizer files included.
- vLLM: <https://github.com/vllm-project/vllm> — inference engine.
- NumPy: <https://numpy.org> — CPU numerical analysis dependency.
- tau²-bench: <https://github.com/sierra-research/tau2-bench> — benchmark
  referenced only in the earlier-study context.

Consult each upstream project and the exact source dataset release for
its license. This snapshot does not grant rights over upstream assets.
