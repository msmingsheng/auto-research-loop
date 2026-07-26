"""
Rule-based grader for chart-type selection: given the data returned by the
SQL step, did the agent pick an appropriate chart type? This is deterministic
-- "line chart for a trend" isn't a judgment call, it's a fixed rule -- so
unlike insight quality, this doesn't need an LLM judge at all.
"""
from dataclasses import dataclass

ALLOWED_CHART_TYPES = {
    "time_series": {"line", "area"},
    "categorical_comparison": {"bar", "column"},
    "part_of_whole": {"pie", "donut", "stacked_bar"},
}


@dataclass
class GradeResult:
    passed: bool
    explanation: str


def grade(chart_type: str, data_shape: str) -> GradeResult:
    chart_type = chart_type.strip().lower().split()[0] if chart_type.strip() else ""
    allowed = ALLOWED_CHART_TYPES.get(data_shape, set())
    if chart_type in allowed:
        return GradeResult(True, f"'{chart_type}' is appropriate for {data_shape} data.")
    return GradeResult(
        False,
        f"Agent chose '{chart_type}' but data_shape is '{data_shape}'; "
        f"expected one of {sorted(allowed)}.",
    )


def grade_suite(tasks: list, engine, system_prompt: str) -> dict:
    results = []
    for task in tasks:
        agent_input = f"[CHART_SELECTION] Question: {task['question']}\nData: {task['data']}"
        candidate = engine.generate(agent_input, system=system_prompt)
        result = grade(candidate, task["data_shape"])  # data_shape is the hidden gold label
        results.append({"task_id": task["id"], "question": task["question"],
                         "candidate_output": candidate, "result": result})

    n_pass = sum(1 for r in results if r["result"].passed)
    return {
        "score": n_pass / len(tasks) if tasks else 0.0,
        "n_pass": n_pass,
        "n_total": len(tasks),
        "details": results,
    }
