"""
Defines every optimizable skill: its task data, its grader, and where its
prompt lives on disk. This is the one file to edit when adding a new skill --
nothing in the optimization loop itself needs to know skill-specific details.

Each skill's grade_suite_fn must have signature (tasks, agent_engine, prompt)
-> {"score", "n_pass", "n_total", "details": [{"task_id","question",
"candidate_output","result": <has .passed, .explanation>}, ...]}.
Graders that need something extra (like the insight judge needing a judge
engine) get it bound in here via functools.partial, so the loop never has
to special-case them.
"""
import functools
import hashlib
import json
import os

from db.setup_db import DB_PATH
from graders import sql_execution_grader, chart_rule_grader, llm_judge_insight

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def _text2sql_db_path():
    """Use an explicit override, then the snapshot recorded by the generator."""
    override = os.environ.get("ANALYTICS_DB_PATH")
    if override:
        return os.path.abspath(override)
    manifest_path = os.path.join(DATA_DIR, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        recorded = manifest.get("db_path")
        if recorded:
            if not os.path.exists(recorded):
                raise FileNotFoundError(
                    f"dataset database snapshot is missing: {recorded}. "
                    "Restore it or set ANALYTICS_DB_PATH explicitly."
                )
            expected_hash = manifest.get("db_sha256")
            if expected_hash:
                digest = hashlib.sha256()
                with open(recorded, "rb") as db_file:
                    for block in iter(lambda: db_file.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_hash:
                    raise RuntimeError(
                        "dataset database snapshot changed after ground truth was generated; "
                        "regenerate the dataset or restore the frozen snapshot"
                    )
            return recorded
    return DB_PATH


def get_skills(optimizer_engine):
    return [
        dict(
            name="text2sql",
            prompt_path=os.path.join(PROMPTS_DIR, "text2sql_prompt.txt"),
            train_tasks=_load("tasks_train"),
            heldout_tasks=_load("tasks_heldout"),
            grade_suite_fn=functools.partial(sql_execution_grader.grade_suite,
                                             _text2sql_db_path()),
        ),
        dict(
            name="chart_selection",
            prompt_path=os.path.join(PROMPTS_DIR, "chart_prompt.txt"),
            train_tasks=_load("tasks_charts_train"),
            heldout_tasks=_load("tasks_charts_heldout"),
            grade_suite_fn=chart_rule_grader.grade_suite,
        ),
        dict(
            name="insight_generation",
            prompt_path=os.path.join(PROMPTS_DIR, "insight_prompt.txt"),
            train_tasks=_load("tasks_insights_train"),
            heldout_tasks=_load("tasks_insights_heldout"),
            grade_suite_fn=functools.partial(
                llm_judge_insight.grade_suite, judge_engine=optimizer_engine
            ),
        ),
    ]
