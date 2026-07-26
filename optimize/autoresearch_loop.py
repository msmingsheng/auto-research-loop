"""
Autonomous overnight loop, in the spirit of karpathy/autoresearch:
fixed number of trials per skill, each trial produces a candidate prompt,
we score it against that skill's FROZEN heldout set, keep it only if it
beats the current best, log every trial's transcript.

Runs all three skills (text2sql, chart_selection, insight_generation) in
one pass -- each optimizes its own prompt file independently, using its
own grader, but shares the same optimizer model.

Run: python optimize/autoresearch_loop.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.llm_engine import get_optimizer_engine, MockAgentEngine  # noqa: E402
from engine.agent_adapter import ObjectAgentAdapter, EndpointAgentAdapter  # noqa: E402
from optimize.text_optimizer import PromptVariable, TGDLite, collect_gradients  # noqa: E402
from optimize.skills import get_skills  # noqa: E402
from db.setup_db import build_db  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
N_TRIALS = 5


# ============================================================================
# CONFIGURE YOUR AGENT HERE -- the only thing you need to change to point
# the harness at your real agent instead of the offline mock.
#
# If your agent handles all three skills through one method (with the
# skill-specific system prompt doing the differentiation), one adapter
# instance for all three skills is fine -- that's the default below.
#
# If your agent has separate methods per skill (e.g. agent.write_sql(),
# agent.pick_chart(), agent.summarize()), return a dict instead so each
# skill gets routed to the right one -- see the commented example.
# ============================================================================
def get_agent_engines(skill_names):
    # --- Option A: one agent object/endpoint handles all skills -------------
    # from my_project.agent import AnalyticsAgent
    # my_agent = AnalyticsAgent(...)
    # engine = ObjectAgentAdapter(my_agent, method="run")
    # return {name: engine for name in skill_names}

    # --- Option B: different method per skill --------------------------------
    # from my_project.agent import AnalyticsAgent
    # my_agent = AnalyticsAgent(...)
    # return {
    #     "text2sql": ObjectAgentAdapter(my_agent, method="write_sql"),
    #     "chart_selection": ObjectAgentAdapter(my_agent, method="pick_chart"),
    #     "insight_generation": ObjectAgentAdapter(my_agent, method="summarize"),
    # }

    # --- Default: offline mock, for testing the harness itself --------------
    mock = MockAgentEngine()
    return {name: mock for name in skill_names}
# ============================================================================


def log_trial(skill_name, trial_idx, prompt_value, train_score, heldout_score, kept):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"trials_{skill_name}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({
            "trial": trial_idx,
            "timestamp": time.time(),
            "prompt": prompt_value,
            "train_score": train_score,
            "heldout_score": heldout_score,
            "kept": kept,
        }) + "\n")


def optimize_skill(skill, agent_engine, optimizer_engine):
    name = skill["name"]
    with open(skill["prompt_path"]) as f:
        current_prompt = f.read().strip()

    prompt_var = PromptVariable(value=current_prompt, role_description=f"{name} system prompt")
    optimizer = TGDLite(optimizer_engine, parameters=[prompt_var])
    grade_suite_fn = skill["grade_suite_fn"]

    best_prompt = prompt_var.value
    best_heldout_score = grade_suite_fn(skill["heldout_tasks"], agent_engine, best_prompt)["score"]
    print(f"[{name}] baseline heldout score = {best_heldout_score:.2f}")
    log_trial(name, 0, best_prompt, None, best_heldout_score, kept=True)

    for trial in range(1, N_TRIALS + 1):
        prompt_var.value = best_prompt
        prompt_var.zero_grad()

        train_result = collect_gradients(
            agent_engine, optimizer_engine, prompt_var, grade_suite_fn, skill["train_tasks"]
        )
        train_score = train_result["score"]

        optimizer.step()
        candidate_prompt = prompt_var.value

        heldout_score = grade_suite_fn(skill["heldout_tasks"], agent_engine, candidate_prompt)["score"]

        kept = heldout_score > best_heldout_score
        print(f"[{name}] trial {trial}: train={train_score:.2f} heldout={heldout_score:.2f} "
              f"{'KEPT' if kept else 'discarded'}")
        log_trial(name, trial, candidate_prompt, train_score, heldout_score, kept)

        if kept:
            best_prompt = candidate_prompt
            best_heldout_score = heldout_score

    with open(skill["prompt_path"], "w") as f:
        f.write(best_prompt + "\n")

    print(f"[{name}] final heldout score: {best_heldout_score:.2f} -> {skill['prompt_path']}\n")
    return best_heldout_score


def main():
    build_db(overwrite=True)  # fresh, deterministic DB each run

    optimizer_engine = get_optimizer_engine()
    skills = get_skills(optimizer_engine)
    agent_engines = get_agent_engines([s["name"] for s in skills])

    print(f"[engines] optimizer={type(optimizer_engine).__name__}")
    for s in skills:
        print(f"          {s['name']} agent={type(agent_engines[s['name']]).__name__}")
    print()

    final_scores = {}
    for skill in skills:
        final_scores[skill["name"]] = optimize_skill(
            skill, agent_engines[skill["name"]], optimizer_engine
        )

    print("=== Summary ===")
    for name, score in final_scores.items():
        print(f"  {name}: {score:.2f}")
    print(f"Full per-skill trial logs in {LOG_DIR}/trials_<skill>.jsonl -- read them before trusting this.")


if __name__ == "__main__":
    main()
