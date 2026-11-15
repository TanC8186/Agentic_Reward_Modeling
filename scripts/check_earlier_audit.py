"""Reconcile anonymous histograms of the earlier 37-pool audit; no scoring."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(summary):
    if summary["pool_count"] != 37 or summary["candidate_count"] != 370 or summary["candidates_per_pool"] != 10:
        raise ValueError("Earlier-study denominator changed")
    if summary["pool_count"] * summary["candidates_per_pool"] != summary["candidate_count"]:
        raise ValueError("Candidate/pool reconciliation failed")
    rates = {}
    for prefix, expected_conditions in (("process_repeat", 5), ("serialization", 3)):
        if summary[prefix + "_count"] != expected_conditions:
            raise ValueError("Condition count changed")
        if len(set(summary[prefix + "_conditions"])) != expected_conditions:
            raise ValueError("Condition labels duplicated or missing")
        histogram = summary[prefix + "_winner_histogram"]
        if len({row["distinct_selected_winners"] for row in histogram}) != len(histogram):
            raise ValueError("Duplicate histogram bin")
        for row in histogram:
            if not (isinstance(row["pool_count"], int) and row["pool_count"] >= 0 and
                    isinstance(row["distinct_selected_winners"], int) and
                    1 <= row["distinct_selected_winners"] <= expected_conditions):
                raise ValueError("Invalid histogram value")
        if sum(row["pool_count"] for row in histogram) != summary["pool_count"]:
            raise ValueError("Histogram does not retain all 37 pools")
        same = sum(row["pool_count"] for row in histogram if row["distinct_selected_winners"] == 1)
        rates[prefix] = {"same_winner_pools": same, "total_pools": summary["pool_count"], "agreement": same / summary["pool_count"]}
    if rates["process_repeat"]["same_winner_pools"] != 37 or rates["serialization"]["same_winner_pools"] != 21:
        raise ValueError("Retained agreement counts changed")
    if not math.isclose(rates["serialization"]["agreement"], summary["reported_serialization_winner_agreement"], abs_tol=1e-12):
        raise ValueError("Reported serialization rate disagrees with histogram")
    if summary["reported_process_repeat_maximum_score_drift"] != 0:
        raise ValueError("Retained reported drift changed")
    conditions = summary["serialization_selection_summaries"]
    if {row["condition"] for row in conditions} != set(summary["serialization_conditions"]):
        raise ValueError("Condition summary mismatch")
    for row in conditions:
        if row["n_pools"] != 37 or not 0 <= row["reported_success_at_1"] <= 1:
            raise ValueError("Invalid condition denominator or utility")
        if not math.isclose(row["reported_success_at_1"] - row["reported_excess_utility"], .5, abs_tol=1e-12):
            raise ValueError("Condition utility/excess summary mismatch")
    return {"status": "PASS", "study": summary["study"], "candidate_count": summary["candidate_count"], **rates, "tier": "retained histogram arithmetic; no model re-scoring or score-drift reconstruction"}


def main():
    summary = json.loads((ROOT / "data/earlier_audit_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(check(summary), indent=2))


if __name__ == "__main__":
    main()
