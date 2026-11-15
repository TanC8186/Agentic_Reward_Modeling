"""Recompute compact development evidence; no model calls and no file writes."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from run_eswa_task_evidence import bootstrap_difference, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recompute(rows, reference):
    expected_counts = {"A": 61, "B": 163, "C": 163, "D": 163, "E": 163}
    require(Counter(row["arm"] for row in rows) == expected_counts, "Task-arm denominators differ from the retained development record")
    paired = defaultdict(dict)
    for row in rows:
        require(row["arm"] not in paired[row["task_id"]], "Duplicate task-arm row")
        paired[row["task_id"]][row["arm"]] = row
        require(row["route"] in ("full", "chunked"), "Unknown route")
        for name in ("utility", "uniform_utility", "oracle"):
            require(isinstance(row[name], (int, float)) and math.isfinite(row[name]) and 0 <= row[name] <= 1, f"Invalid {name}")
        require(isinstance(row["top_set_size"], int) and 1 <= row["top_set_size"] <= 3, "Invalid top-set size")
        require(isinstance(row["uniform_fallback"], bool) and isinstance(row["mixed"], bool), "Invalid flag")
    require(len(paired) == 163, "Expected 163 independent development tasks")
    require(len({r["intent_template_id"] for r in rows}) == 38, "Expected 38 task-template groups")
    for arms in paired.values():
        require({"B", "C", "D", "E"} <= arms.keys(), "Missing paired comparison arm")
        base = arms["D"]
        require(("A" in arms) == (base["route"] == "full"), "Full-only A coverage differs")
        for row in arms.values():
            for name in ("intent_template_id", "route", "uniform_utility", "oracle", "mixed"):
                require(row[name] == base[name], f"Inconsistent paired field: {name}")
    means = {}
    for arm in expected_counts:
        arm_rows = [r for r in rows if r["arm"] == arm]
        means[arm] = sum(r["utility"] for r in arm_rows) / len(arm_rows)
        expected = next(r["all"]["success"] for r in reference["arms"] if r["arm"] == arm)
        require(math.isclose(means[arm], expected, rel_tol=0, abs_tol=1e-12), f"Mean changed for {arm}")
    primary = bootstrap_difference(rows)
    expected = reference["primary_E_minus_D"]
    for name in ("tasks", "template_groups", "tasks_improved", "tasks_harmed", "bootstrap_repeats", "seed"):
        require(primary[name] == expected[name], f"Primary analysis field changed: {name}")
    for observed, retained in zip([primary["difference"], *primary["ci95"]], [expected["difference"], *expected["ci95"]]):
        require(math.isclose(observed, retained, rel_tol=0, abs_tol=1e-12), "Primary estimate or interval changed")
    return {"status": "PASS", "tier": "retained-output analysis; no model re-scoring", "rows": len(rows), "tasks": len(paired), "template_groups": 38, "success_by_arm": means, "primary_E_minus_D": primary, "evidence_status": "ANALYZED_development_only"}


def main():
    rows = read_jsonl(ROOT / "data/development_pool_utilities.jsonl")
    reference = json.loads((ROOT / "data/development_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(recompute(rows, reference), indent=2))


if __name__ == "__main__":
    main()
