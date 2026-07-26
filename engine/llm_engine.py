"""
OPTIMIZER engine only -- the stronger model that grades failures and
rewrites the agent's prompt (e.g. Claude Opus 4.8).

The AGENT (the thing under test) is not selected here at all. Bring your
own agent via engine/agent_adapter.py -- either wrap an object you already
have, or an HTTP endpoint it runs behind. See autoresearch_loop.py's
CONFIGURE YOUR AGENT section.
"""
import os


class ClaudeEngine:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OpenAIEngine:
    def __init__(self, model: str):
        import openai
        self.client = openai.OpenAI()
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()


class MockOptimizerEngine:
    """
    Offline stand-in for the stronger grading/optimizing model.
    Three jobs across the three skills:
      1. turn a grader's failure explanation into specific critique
      2. rewrite a skill's prompt given accumulated critique
      3. act as the LLM judge for insight quality (GROUNDED/RELEVANT rubric)
    """

    def generate(self, prompt: str, system: str = "") -> str:
        p = prompt.lower()
        if "grounded: yes/no" in p:
            return self._judge_insight(prompt)
        if "rewrite the prompt" in p:
            return self._rewrite_prompt(p)
        return self._critique_failure(p)

    @staticmethod
    def _rewrite_prompt(p: str) -> str:
        if "chart" in p:
            return (
                "You are a chart-selection assistant. Given a question and the data "
                "returned for it, choose the best chart type.\n"
                "Use a line chart for time-based or trend data, a pie chart for "
                "part-to-whole or percentage-of-total breakdowns, and a bar chart for "
                "categorical comparisons otherwise.\n"
                "Return only the chart type (one word), nothing else."
            )
        if "insight" in p:
            return (
                "You are an analytics assistant. Given a question and the data "
                "returned for it, write a short insight summarizing what the data shows.\n"
                "Only state numbers and figures that appear in the provided data -- "
                "do not invent figures. Directly address the question asked, referencing "
                "the specific data given."
            )
        if "join" in p:
            return (
                "You are a SQL assistant for a sales analytics database. "
                "Given a schema and a question, write a SQL query that answers it.\n"
                "Always use explicit JOIN clauses (not implicit comma joins) and make sure "
                "every table needed to compute the requested metric is joined in -- "
                "for revenue questions this means joining orders, products, customers, "
                "and regions as needed so unit_price is available.\n"
                "Return only the SQL query, nothing else."
            )
        return ""  # no actionable pattern recognized yet -- signals no-op to TGDLite

    @staticmethod
    def _critique_failure(p: str) -> str:
        # dispatch by the SHAPE of the grader's explanation, not the skill name,
        # so this stays generic to whatever grader produced it
        if "grounded:" in p:
            return ("The insight includes claims or numbers not present in the given data, "
                     "or doesn't directly address the question. The prompt should require "
                     "the agent to only use numbers from the provided data and directly "
                     "answer the question asked.")
        if "data_shape is" in p:
            return ("The chosen chart type doesn't match the data's shape. The prompt should "
                     "specify which chart type to use for time-based/trend data vs. "
                     "part-to-whole/percentage data vs. plain categorical comparisons.")
        if "execution failed" in p:
            return ("The query failed to execute -- check table/column names and use "
                     "explicit JOIN syntax rather than implicit comma joins.")
        return ("The result set is wrong, most likely because a required table "
                "(e.g. products, for unit_price) was never joined in before aggregating. "
                "The prompt should explicitly require joining every table needed to "
                "compute the requested metric.")

    @staticmethod
    def _judge_insight(prompt: str) -> str:
        import re
        data_match = re.search(r"data returned \(ground truth\):\s*(.*)", prompt, re.IGNORECASE)
        insight_match = re.search(r"insight produced by the agent:\s*(.*)", prompt, re.IGNORECASE)
        data_str = data_match.group(1) if data_match else ""
        insight_str = insight_match.group(1) if insight_match else ""

        data_numbers = set(re.findall(r"\d+(?:\.\d+)?", data_str))
        insight_numbers = set(re.findall(r"\d+(?:\.\d+)?", insight_str))
        grounded = bool(data_numbers) and data_numbers.issubset(insight_numbers)

        question_match = re.search(r"question asked:\s*(.*)", prompt, re.IGNORECASE)
        question_str = question_match.group(1) if question_match else ""
        stopwords = {"what", "does", "the", "show", "tell", "us", "can", "we", "from", "about"}
        q_words = {w.strip("?.,").lower() for w in question_str.split()
                   if len(w) > 3 and w.lower() not in stopwords}
        relevant = any(w in insight_str.lower() for w in q_words) if q_words else True

        reason = "OK" if (grounded and relevant) else (
            "Insight contains figures not present in the data, or invents generic claims."
            if not grounded else "Insight doesn't directly reference the question's subject."
        )
        return f"GROUNDED: {'yes' if grounded else 'no'}\nRELEVANT: {'yes' if relevant else 'no'}\nREASON: {reason}"


class MockAgentEngine:
    """
    Offline stand-in for YOUR agent, used only when no real agent/endpoint
    is configured yet -- lets you test the harness itself before wiring
    in the real thing. Simulates a model that:
      - forgets to join in unit_price for revenue SQL questions
      - defaults to 'bar' charts regardless of data shape
      - writes generic, ungrounded insights that ignore the actual data
    until each skill's prompt explicitly tells it not to.
    """

    def generate(self, prompt: str, system: str = "") -> str:
        if prompt.startswith("[CHART_SELECTION]"):
            return self._chart(prompt, system)
        if prompt.startswith("[INSIGHT]"):
            return self._insight(prompt, system)
        return self._sql(prompt, system)

    def _sql(self, prompt: str, system: str) -> str:
        q = prompt.lower()
        wants_join = "join" in system.lower() and "explicit" in system.lower()

        if "total revenue" in q and "region" in q:
            if wants_join:
                return (
                    "SELECT r.region_name, SUM(o.quantity * p.unit_price) AS revenue "
                    "FROM orders o "
                    "JOIN customers c ON o.customer_id = c.customer_id "
                    "JOIN products p ON o.product_id = p.product_id "
                    "JOIN regions r ON c.region_id = r.region_id "
                    "GROUP BY r.region_name;"
                )
            return (
                "SELECT r.region_name, SUM(o.quantity) AS revenue "
                "FROM orders o "
                "JOIN customers c ON o.customer_id = c.customer_id "
                "JOIN regions r ON c.region_id = r.region_id "
                "GROUP BY r.region_name;"
            )

        if "top" in q and "product" in q:
            if wants_join:
                return (
                    "SELECT p.product_name, SUM(o.quantity) AS units "
                    "FROM orders o JOIN products p ON o.product_id = p.product_id "
                    "GROUP BY p.product_name ORDER BY units DESC LIMIT 1;"
                )
            return "SELECT product_id, SUM(quantity) FROM orders GROUP BY product_id;"

        return "SELECT 1;"

    def _chart(self, prompt: str, system: str) -> str:
        q = prompt.lower()
        wants_shape_aware = "line chart" in system.lower() and "pie chart" in system.lower()
        if wants_shape_aware:
            if "over the last" in q or "trend" in q or "monthly" in q or "over time" in q:
                return "line"
            if "percentage of" in q or "share of" in q or "breakdown" in q:
                return "pie"
            return "bar"
        return "bar"  # naive baseline: always guesses bar regardless of shape

    def _insight(self, prompt: str, system: str) -> str:
        wants_grounded = "do not invent" in system.lower() or "only state numbers" in system.lower()
        question = prompt.split("Question:")[1].split("\nData:")[0].strip() if "Question:" in prompt else ""
        data_str = prompt.split("Data:")[1].strip() if "Data:" in prompt else ""
        if wants_grounded and data_str:
            # weaves the actual question text in too, so a correctly-prompted
            # agent is both grounded (uses only given numbers) and relevant
            # (actually addresses what was asked)
            return f"To answer '{question}': {data_str}"
        return "Overall performance looks strong and trending upward across the board."


def get_optimizer_engine():
    """The stronger model doing grading/critique/prompt-rewriting, e.g. Opus 4.8."""
    provider = os.environ.get("OPTIMIZER_PROVIDER", "").lower()
    model = os.environ.get("OPTIMIZER_MODEL", "claude-opus-4-8")
    if provider == "openai":
        return OpenAIEngine(model)
    if provider == "anthropic" or (not provider and os.environ.get("ANTHROPIC_API_KEY")):
        return ClaudeEngine(model)
    return MockOptimizerEngine()
