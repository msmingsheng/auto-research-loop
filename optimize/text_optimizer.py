"""
A minimal TextGrad-style optimizer: Variable + textual "backward pass" +
step(), following the same shape as zou-group/textgrad's Variable / TextLoss
/ TGD API -- implemented locally so it runs offline against the mock engines.

Deliberately takes TWO engines, never one:
  - agent_engine: the weaker model actually performing the skill (forward pass)
  - optimizer_engine: the stronger model grading failures and rewriting
    the prompt (backward pass / step)
This mirrors how you'd actually run it: GPT-5.1 as the agent under test,
Opus 4.8 doing the grading and prompt rewriting.
"""
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class PromptVariable:
    value: str
    role_description: str
    gradients: List[str] = field(default_factory=list)

    def zero_grad(self):
        self.gradients = []


class TGDLite:
    """Textual Gradient Descent: rewrite the prompt using accumulated critique."""

    def __init__(self, optimizer_engine, parameters: List[PromptVariable]):
        self.optimizer_engine = optimizer_engine
        self.parameters = parameters

    def step(self):
        for p in self.parameters:
            if not p.gradients:
                continue
            critique = "\n".join(f"- {g}" for g in p.gradients)
            meta_prompt = (
                f"You are improving a prompt with this role: {p.role_description}\n\n"
                f"Current prompt:\n{p.value}\n\n"
                f"Feedback from grading failed cases (produced by a weaker model "
                f"running under this prompt):\n{critique}\n\n"
                "Rewrite the prompt to fix these failure patterns. "
                "Be specific and directive. Return only the new prompt text."
            )
            new_value = self.optimizer_engine.generate(
                meta_prompt, system="You are a senior prompt engineer optimizing "
                                     "instructions for a weaker downstream model."
            )
            if new_value.strip():
                p.value = new_value.strip()
            p.zero_grad()


def collect_gradients(agent_engine, optimizer_engine, prompt_var: PromptVariable,
                       grade_suite_fn: Callable, tasks: list):
    """
    Skill-agnostic. Forward pass: run the TRAIN suite with the (weaker)
    agent_engine under the current prompt, using whichever grader this
    skill was configured with (execution-match, rule-based, or LLM-judge --
    collect_gradients doesn't need to know which). Backward pass: for every
    failing task, ask the (stronger) optimizer_engine to turn the grader's
    explanation into specific, actionable critique -- the textual analogue
    of a gradient.
    """
    suite_result = grade_suite_fn(tasks, agent_engine, prompt_var.value)
    for detail in suite_result["details"]:
        r = detail["result"]
        if r.passed:
            continue
        critique_prompt = (
            f"A weaker model was asked this question: {detail['question']}\n"
            f"It produced this output: {detail['candidate_output']}\n"
            f"Grader found: {r.explanation}\n\n"
            "Explain concisely and specifically what instruction is missing from "
            "the prompt that would have prevented this exact mistake."
        )
        fb = optimizer_engine.generate(
            critique_prompt, system="You are a rigorous grading expert."
        )
        prompt_var.gradients.append(f"Q: {detail['question']} -> {fb}")
    return suite_result
