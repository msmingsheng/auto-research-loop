"""
LLM-as-judge grader for insight quality. Graded by the OPTIMIZER engine
(the strong model), never by the agent under test -- an agent grading its
own output is not a grader.

Two rubric criteria, both must pass:
  - GROUNDED: every number/claim in the insight must trace back to the
    data actually given, no invented figures.
  - RELEVANT: the insight must actually answer the question asked.

This is inherently fuzzier than execution-match or rule-based grading --
periodically spot-check a sample of judge verdicts against human judgment
(see README caution on LLM-judge drift).
"""
from dataclasses import dataclass

RUBRIC_PROMPT_TEMPLATE = """You are grading an analytics insight for two criteria.

Question asked: {question}
Data returned (ground truth): {data}
Insight produced by the agent: {insight}

Criteria:
1. GROUNDED: every specific number or claim in the insight must be traceable to the data above. No invented figures.
2. RELEVANT: the insight must actually answer the question asked, not a tangential observation.

Respond in exactly this format:
GROUNDED: yes/no
RELEVANT: yes/no
REASON: one sentence explaining any failure
"""


@dataclass
class GradeResult:
    passed: bool
    explanation: str


def grade(question: str, data: str, insight: str, judge_engine) -> GradeResult:
    prompt = RUBRIC_PROMPT_TEMPLATE.format(question=question, data=data, insight=insight)
    raw = judge_engine.generate(prompt, system="You are a strict, precise grading assistant.")
    grounded = "grounded: yes" in raw.lower()
    relevant = "relevant: yes" in raw.lower()
    return GradeResult(passed=(grounded and relevant), explanation=raw.strip())


def grade_suite(tasks: list, agent_engine, system_prompt: str, judge_engine) -> dict:
    results = []
    for task in tasks:
        agent_input = f"[INSIGHT] Question: {task['question']}\nData: {task['data']}"
        insight = agent_engine.generate(agent_input, system=system_prompt)
        result = grade(task["question"], task["data"], insight, judge_engine)
        results.append({"task_id": task["id"], "question": task["question"],
                         "candidate_output": insight, "result": result})

    n_pass = sum(1 for r in results if r["result"].passed)
    return {
        "score": n_pass / len(tasks) if tasks else 0.0,
        "n_pass": n_pass,
        "n_total": len(tasks),
        "details": results,
    }
