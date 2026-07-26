"""
Execution-accuracy grader for text2sql, in the spirit of Spider/BIRD:
run the candidate SQL against the real DB and diff the *result set*
against the gold query's result set. Never string-match SQL text --
there are many correct ways to write the same query.
"""
import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass
class GradeResult:
    passed: bool
    execution_error: Optional[str]
    row_diff: bool
    explanation: str


def _run(db_path: str, sql: str):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


def grade(db_path: str, candidate_sql: str, gold_sql: str) -> GradeResult:
    cand_rows, cand_err = _run(db_path, candidate_sql)
    if cand_err:
        return GradeResult(
            passed=False,
            execution_error=cand_err,
            row_diff=False,
            explanation=f"Execution failed: {cand_err}",
        )

    gold_rows, gold_err = _run(db_path, gold_sql)
    if gold_err:
        # gold query itself is broken -- a dataset bug, not a model bug
        raise RuntimeError(f"Gold SQL failed to execute: {gold_err}\nSQL: {gold_sql}")

    # order-insensitive compare (most analytics questions don't care about row order
    # unless the question explicitly asks for a ranking/order)
    cand_set = sorted(map(tuple, cand_rows))
    gold_set = sorted(map(tuple, gold_rows))

    if cand_set == gold_set:
        return GradeResult(True, None, False, "Matched gold result set.")

    return GradeResult(
        passed=False,
        execution_error=None,
        row_diff=True,
        explanation=(
            f"Result mismatch. Got {cand_set[:5]}{'...' if len(cand_set) > 5 else ''}, "
            f"expected {gold_set[:5]}{'...' if len(gold_set) > 5 else ''}."
        ),
    )


def grade_suite(db_path: str, tasks: list, engine, system_prompt: str) -> dict:
    """Run a whole task suite against one prompt version, return aggregate + per-task results."""
    results = []
    for task in tasks:
        candidate_sql = engine.generate(task["question"], system=system_prompt)
        result = grade(db_path, candidate_sql, task["gold_sql"])
        results.append({"task_id": task["id"], "question": task["question"],
                         "candidate_output": candidate_sql, "result": result})

    n_pass = sum(1 for r in results if r["result"].passed)
    return {
        "score": n_pass / len(tasks) if tasks else 0.0,
        "n_pass": n_pass,
        "n_total": len(tasks),
        "details": results,
    }
