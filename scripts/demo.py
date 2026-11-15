"""Run the original renderer/prompt builder on a synthetic record, without an LLM."""
import json

from run_eswa_task_evidence import note_prompt, render_events, render_webarena


def synthetic_record():
    context = {"type": "human", "content": "SYNTHETIC task: set the demo basket quantity to two."}
    page = {"type": "human", "content": "[Current state starts here] demo basket quantity: 1"}
    return {"candidate_id": "synthetic-candidate", "task_id": "synthetic-task", "site": "synthetic", "task_prompt": "Set the demo basket quantity to two.", "steps": {
        "0": {"input_messages": {"contents": [context, page]}, "output_messages": {"tool_call_message": {"content": "Set quantity to two"}, "tool_response": {"content": "Action accepted; final page not yet observed"}}},
        "1": {"input_messages": {"contents": [context, page]}, "output_messages": {"tool_call_message": {"content": "Claim: completed"}}},
        "2": {"output_messages": {"tool_response": {"content": "Source input observation missing"}}},
    }}


def main():
    row = render_webarena(synthetic_record())
    transcript = render_events(row["events"])
    print(json.dumps({"example_type": "SYNTHETIC; no model calls or benchmark result", "source_steps": row["source_steps"], "repeated_packaging_messages_removed": row["snapshot_messages_removed"], "current_page_observations_retained": sum(e["source"] == "environment_observation_before_action" for e in row["events"]), "missing_source_input_steps": row["missing_input_steps"], "interpretation": "Both page observations still show quantity one; an accepted action or the agent's completion claim is not an observed final state.", "structured_E_prompt": note_prompt(row, "G1: final demo basket quantity is two", "E", transcript, "", 0, 1)}, indent=2))


if __name__ == "__main__":
    main()
